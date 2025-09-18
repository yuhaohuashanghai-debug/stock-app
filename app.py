import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

st.set_page_config(page_title="📈 股票&ETF AI分析平台", layout="wide")
st.title("📊 单只&批量选股 · 技术分析 · 回测 · AI风控解读")

# ================== 控制面板 ==================
with st.sidebar:
    st.header("⚙️ 控制面板")
    mode = st.radio("分析模式", ["单只分析", "批量选股&回测"], horizontal=True)
    code_type = st.radio("类型", ["A股", "ETF"], horizontal=True)
    # AI KEY 统一入口
    DEEPSEEK_API_KEY = st.text_input("DeepSeek API Key（选填，用于AI风控）", type="password")

# ========== 数据接口 ==========
@st.cache_data(ttl=300)
def fetch_realtime_kline(code: str, code_type: str):
    try:
        if code_type == "A股":
            symbol = f"sh{code}" if code.startswith("6") else f"sz{code}"
            df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        else:
            df = pd.DataFrame()
            for etf_func in [
                lambda c: ak.fund_etf_hist_sina(symbol=c),
                lambda c: ak.fund_etf_hist_em(symbol=c),
                lambda c: ak.fund_etf_hist_jsl(symbol=c)
            ]:
                try:
                    tmp = etf_func(code)
                    if tmp is not None and not tmp.empty:
                        df = tmp
                        break
                except Exception:
                    continue
        df = df.reset_index(drop=True)
        name_map = {
            "date": "date", "日期": "date", "交易日期": "date",
            "open": "open", "开盘": "open",
            "close": "close", "收盘": "close",
            "high": "high", "最高": "high",
            "low": "low", "最低": "low",
            "volume": "volume", "成交量": "volume", "成交量(手)": "volume", "成交量(股)": "volume"
        }
        df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
        need_cols = ["date", "open", "close", "high", "low", "volume"]
        miss = [x for x in need_cols if x not in df.columns]
        if miss:
            return None
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_stock_news(code: str, code_type: str):
    try:
        if code_type == "ETF":
            return ["ETF暂无个股新闻"]
        df = ak.stock_news_em(symbol=code)
        for col in ["title", "新闻标题", "标题"]:
            if col in df.columns:
                return df[col].head(5).tolist()
        return ["未找到新闻标题字段"]
    except Exception as e:
        return [f"新闻获取失败: {e}"]

@st.cache_data(ttl=300)
def fetch_fund_flow(code: str, code_type: str):
    try:
        if code_type == "A股":
            df = ak.stock_individual_fund_flow(stock=code)
            df = df.tail(5).reset_index(drop=True)
            for col in ["主力净流入-净额", "主力净流入", "主力净流入净额", "主力资金流入", "主力资金净流入"]:
                if col in df.columns:
                    return df[["日期", col]].rename(columns={col: "主力净流入"}).to_dict("records")
            return [{"error": f"未找到主力净流入字段，现有字段: {df.columns.tolist()}"}]
        else:
            df = ak.fund_etf_hist_em(symbol=code)
            df = df.tail(5)
            if "日期" in df.columns and "成交额" in df.columns and "成交量" in df.columns:
                return df[["日期", "成交额", "成交量"]].to_dict("records")
            else:
                return [{"error": f"ETF接口无成交额/量字段，返回: {df.columns.tolist()}"}]
    except Exception as e:
        return [{"error": str(e)}]

def format_money(x):
    try:
        x = float(x)
        if abs(x) >= 1e8:
            return f"{x/1e8:.2f} 亿"
        elif abs(x) >= 1e4:
            return f"{x/1e4:.2f} 万"
        else:
            return f"{x:.0f}"
    except:
        return str(x)

def add_indicators(df: pd.DataFrame, indicator: str):
    df["MA5"] = ta.sma(df["close"], length=5)
    df["MA20"] = ta.sma(df["close"], length=20)
    if indicator == "MACD":
        macd = ta.macd(df["close"])
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACDh"] = macd["MACDh_12_26_9"]
        df["MACDs"] = macd["MACDs_12_26_9"]
    elif indicator == "RSI":
        df["RSI"] = ta.rsi(df["close"], length=14)
    elif indicator == "BOLL":
        boll = ta.bbands(df["close"], length=20, std=2)
        df["BOLL_U"] = boll["BBU_20_2.0"]
        df["BOLL_M"] = boll["BBM_20_2.0"]
        df["BOLL_L"] = boll["BBL_20_2.0"]
    elif indicator == "KDJ":
        kdj = ta.stoch(df["high"], df["low"], df["close"])
        df["K"] = kdj["STOCHk_14_3_3"]
        df["D"] = kdj["STOCHd_14_3_3"]
        df["J"] = 3 * df["K"] - 2 * df["D"]
    return df.dropna()

def plot_chart(df: pd.DataFrame, code: str, indicator: str, show_ma: list, show_volume: bool):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.05,
                        subplot_titles=(f"{code} K线图", "成交量", indicator))
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线图"
    ), row=1, col=1)
    if "MA5" in show_ma:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], name="MA5"), row=1, col=1)
    if "MA20" in show_ma:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20"), row=1, col=1)
    if show_volume:
        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量", opacity=0.4), row=2, col=1)
    if indicator == "MACD":
        fig.add_trace(go.Bar(x=df["date"], y=df["MACDh"], name="MACDh", opacity=0.3), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], name="信号线"), row=3, col=1)
    elif indicator == "RSI":
        fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI"), row=3, col=1)
    elif indicator == "BOLL":
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_U"], name="上轨"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_M"], name="中轨"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_L"], name="下轨"), row=1, col=1)
    elif indicator == "KDJ":
        fig.add_trace(go.Scatter(x=df["date"], y=df["K"], name="K"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["D"], name="D"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["J"], name="J"), row=3, col=1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, showlegend=True)
    return fig

# ========== 批量选股&回测信号 ==========
def check_signals(df, strategies):
    res = []
    if "MA金叉" in strategies:
        ma5 = ta.sma(df["close"], length=5)
        ma20 = ta.sma(df["close"], length=20)
        if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]:
            res.append("MA金叉")
    if "MACD金叉" in strategies:
        macd = ta.macd(df["close"])
        if macd["MACD_12_26_9"].iloc[-1] > macd["MACDs_12_26_9"].iloc[-1] and \
           macd["MACD_12_26_9"].iloc[-2] <= macd["MACDs_12_26_9"].iloc[-2]:
            res.append("MACD金叉")
    if "RSI超卖反弹" in strategies:
        rsi = ta.rsi(df["close"], length=14)
        if rsi.iloc[-2] < 30 and rsi.iloc[-1] > 30:
            res.append("RSI超卖反弹")
    if "BOLL下轨突破" in strategies:
        boll = ta.bbands(df["close"], length=20)
        if df["close"].iloc[-2] < boll["BBL_20_2.0"].iloc[-2] and df["close"].iloc[-1] > boll["BBL_20_2.0"].iloc[-1]:
            res.append("BOLL下轨突破")
    if "KDJ金叉" in strategies:
        kdj = ta.stoch(df["high"], df["low"], df["close"])
        if kdj["STOCHk_14_3_3"].iloc[-2] < kdj["STOCHd_14_3_3"].iloc[-2] and \
           kdj["STOCHk_14_3_3"].iloc[-1] > kdj["STOCHd_14_3_3"].iloc[-1]:
            res.append("KDJ金叉")
    return res

def backtest_signal(df, signal_func, n_forward=5):
    res = []
    for i in range(len(df) - n_forward):
        if signal_func(df.iloc[:i+1]):
            future_pct = (df["close"].iloc[i+n_forward] - df["close"].iloc[i]) / df["close"].iloc[i] * 100
            res.append(future_pct)
    if res:
        win_rate = sum([1 if x > 0 else 0 for x in res]) / len(res) * 100
        avg_return = sum(res) / len(res)
        return {"信号次数": len(res), f"{n_forward}日均涨跌幅": avg_return, "胜率": win_rate}
    else:
        return {"信号次数": 0, f"{n_forward}日均涨跌幅": 0, "胜率": 0}

def ai_strategy_commentary(pool_df, backtest_df, strategies, api_key):
    stocks_txt = "\n".join([f"{row['code']} {row['signals']}" for _, row in pool_df.iterrows()])
    stats_txt = backtest_df.to_string(index=False)
    prompt = f"""以下是批量选股与回测统计，请以专业投研视角，输出智能策略点评：
【信号策略】：{strategies}
【信号股票池】：\n{stocks_txt}
【信号回测统计】：\n{stats_txt}
请用简明语言总结这些策略的近期选股胜率、风控建议、适合的市场环境、以及操作建议。
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat","messages": [{"role": "user", "content": prompt}],"max_tokens": 512,"temperature": 0.5}
    try:
        resp = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI点评失败：{e}"

# ========== 主程序 ==========
if mode == "单只分析":
    with st.expander("📌 单只基础参数", expanded=True):
        code = st.text_input("股票/ETF代码（如 600519 或 510300）", "600519")
        hold_amount = st.number_input("持有股数", min_value=0, step=100, value=0)
        hold_cost = st.number_input("持仓成本价", min_value=0.0, step=0.01, value=0.0, format="%.2f")
        stop_profit = st.number_input("止盈线（%）", value=10.0, help="如达到该盈利率建议止盈")
        stop_loss = st.number_input("止损线（%）", value=-7.0, help="如达到该亏损率建议止损")
        show_volume = st.checkbox("显示成交量", value=True)
    with st.expander("📊 技术指标设置", expanded=True):
        show_ma = st.multiselect("显示均线", ["MA5", "MA20"], default=["MA5", "MA20"])
        indicator = st.selectbox("选择额外指标", ["MACD", "RSI", "BOLL", "KDJ"])

    analyze_btn = st.button("🚀 单只分析")
    if analyze_btn:
        with st.spinner("数据加载中..."):
            df = fetch_realtime_kline(code, code_type)
            if df is None or df.empty:
                st.error("无可用行情数据")
                st.stop()
            df = add_indicators(df, indicator)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 图表", "📰 新闻", "💰 资金流", "🤖 AI/本地分析"]
        )

        with tab1:
            st.plotly_chart(plot_chart(df, code, indicator, show_ma, show_volume), use_container_width=True)

        with tab2:
            news_list = fetch_stock_news(code, code_type)
            st.subheader("📰 实时消息面")
            for n in news_list:
                st.write("- " + n)

        with tab3:
            fund_flow = fetch_fund_flow(code, code_type)
            if code_type == "A股":
                st.subheader("💰 资金流向（近5日）")
                for f in fund_flow:
                    if "主力净流入" in f:
                        val = format_money(f["主力净流入"])
                        prefix = "+" if f["主力净流入"] > 0 else ""
                        st.write(f"{f['日期']} 主力净流入: {prefix}{val}")
                    elif "error" in f:
                        st.error(f["error"])
                    else:
                        st.write(f)
            else:
                st.subheader("💰 ETF成交额/成交量（近5日）")
                for f in fund_flow:
                    if "成交额" in f and "成交量" in f:
                        st.write(f"{f['日期']} 成交额: {format_money(f['成交额'])}，成交量: {format_money(f['成交量'])}")
                    elif "error" in f:
                        st.error(f["error"])
                    else:
                        st.write(f)

        with tab4:
            latest = df.iloc[-1]
            summary = f"收盘价:{latest['close']:.2f}, MA5:{latest['MA5']:.2f}, MA20:{latest['MA20']:.2f}"
            if indicator == "MACD":
                summary += f", MACD:{latest['MACD']:.3f}, 信号线:{latest['MACDs']:.3f}"
            st.subheader("📌 技术指标总结")
            st.write(summary)
            # 本地浮盈/止盈止损提示
            if hold_amount > 0 and hold_cost > 0:
                pnl = (latest['close'] - hold_cost) * hold_amount
                pnl_rate = (latest['close'] - hold_cost) / hold_cost * 100
                st.write(f"当前持有：{hold_amount} 股，成本价：{hold_cost:.2f}，浮盈：{pnl:.2f} 元，盈亏率：{pnl_rate:.2f}%")
                if pnl_rate >= stop_profit:
                    st.success("【止盈提醒】已达到设定止盈线，建议部分或全部止盈！")
                elif pnl_rate <= stop_loss:
                    st.error("【止损提醒】已触及止损线，建议尽快止损离场！")
                else:
                    st.info("当前未触及止盈/止损线，建议结合AI趋势、技术面再决定。")
            # AI分析
            if DEEPSEEK_API_KEY:
                news_list = fetch_stock_news(code, code_type)
                fund_flow = fetch_fund_flow(code, code_type)
                with st.spinner("DeepSeek AI 概率预测中..."):
                    ai_text = ai_strategy_commentary(
                        pd.DataFrame([{"code": code, "signals": indicator}]),
                        pd.DataFrame(),
                        [indicator],
                        DEEPSEEK_API_KEY
                    )
                    st.subheader("📊 AI 趋势概率+操作建议")
                    st.write(ai_text)
            else:
                st.subheader("🤖 本地技术面/持仓建议")
                if indicator == "MACD":
                    if latest["MACD"] > latest["MACDs"]:
                        st.write("MACD 金叉，短期有反弹可能。")
                    elif latest["MACD"] < latest["MACDs"]:
                        st.write("MACD 死叉，短期下行动能较大。")
                    else:
                        st.write("MACD 持平，市场观望情绪浓。")
                elif indicator == "RSI":
                    if latest["RSI"] < 30:
                        st.write("RSI < 30，超卖区域，可能反弹。")
                    elif latest["RSI"] > 70:
                        st.write("RSI > 70，超买风险，可能回调。")
                    else:
                        st.write("RSI 中性，市场震荡。")

elif mode == "批量选股&回测":
    st.header("📋 批量选股&回测")
    uploaded_file = st.file_uploader("上传股票池Excel（示例见 stocks_example.xlsx）", type=["xlsx"])
    period = st.selectbox("K线周期", ["日线"], index=0)
    strategies = st.multiselect("信号策略", 
        ["MA金叉", "MACD金叉", "RSI超卖反弹", "BOLL下轨突破", "KDJ金叉"], 
        default=["MACD金叉", "RSI超卖反弹"])
    backtest_days = st.selectbox("信号回测未来N日", [3, 5, 10], index=1)
    analyze_btn = st.button("🚀 批量分析", key="batch_analyze")
    if analyze_btn:
        if uploaded_file is None:
            st.warning("请上传Excel股票池文件！")
            st.stop()
        stocks_df = pd.read_excel(uploaded_file)
        if "code" not in stocks_df.columns:
            st.error("Excel必须包含'code'列（股票/ETF代码）")
            st.stop()
        pool_result = []
        st.info("正在批量信号筛选...")
        for idx, row in stocks_df.iterrows():
            code = str(row["code"]).zfill(6)
            df = fetch_realtime_kline(code, code_type)
            if df is None or df.empty or len(df) < 40:
                continue
            sigs = check_signals(df, strategies)
            if sigs:
                pool_result.append({"code": code, "name": row.get("name", ""), "signals": ",".join(sigs)})
        pool_df = pd.DataFrame(pool_result)
        st.subheader("📝 筛选结果")
        if pool_df.empty:
            st.warning("无股票/ETF符合全部选中策略条件")
        else:
            st.dataframe(pool_df)

        st.subheader(f"🔍 回测分析（未来{backtest_days}日表现）")
        backtest_stats = []
        for _, row in pool_df.iterrows():
            code = row["code"]
            df = fetch_realtime_kline(code, code_type)
            def sig_func(sub_df):
                return all([s in check_signals(sub_df, strategies) for s in row["signals"].split(",")])
            bt = backtest_signal(df, sig_func, n_forward=backtest_days)
            backtest_stats.append({"code": code, **bt})
        backtest_df = pd.DataFrame(backtest_stats)
        st.dataframe(backtest_df)

        st.subheader("🤖 AI智能策略点评")
        if not pool_df.empty and DEEPSEEK_API_KEY:
            with st.spinner("AI分析中..."):
                ai_text = ai_strategy_commentary(pool_df, backtest_df, strategies, DEEPSEEK_API_KEY)
                st.write(ai_text)
        else:
            st.write("未配置API KEY，仅展示数据。")

        st.download_button("下载筛选结果", pool_df.to_csv(index=False).encode("utf-8-sig"), "signals_select_result.csv")
        st.download_button("下载回测结果", backtest_df.to_csv(index=False).encode("utf-8-sig"), "signals_backtest_result.csv")
