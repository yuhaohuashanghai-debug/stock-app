import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

st.set_page_config(page_title="📈 股票&ETF AI分析平台", layout="wide")
st.title("📊 实时股票/ETF 技术分析 + 资金流向 + AI趋势/止盈止损建议")

# ========== 控制面板 ==========
with st.sidebar:
    st.header("⚙️ 控制面板")
    with st.expander("📌 基础设置", expanded=True):
        code_type = st.radio("类型", ["A股", "ETF"], horizontal=True)
        code = st.text_input("股票/ETF代码（如 600519 或 510300）", "600519")
        hold_amount = st.number_input("持有股数", min_value=0, step=100, value=0)
        hold_cost = st.number_input("持仓成本价", min_value=0.0, step=0.01, value=0.0, format="%.2f")
        stop_profit = st.number_input("止盈线（%）", value=10.0, help="如达到该盈利率建议止盈")
        stop_loss = st.number_input("止损线（%）", value=-7.0, help="如达到该亏损率建议止损")
        show_volume = st.checkbox("显示成交量", value=True)
    with st.expander("📊 指标设置", expanded=True):
        show_ma = st.multiselect("显示均线", ["MA5", "MA20", "MA30"], default=["MA5", "MA20", "MA30"])
        indicator = st.selectbox("选择额外指标", ["MACD", "RSI", "BOLL", "KDJ"])
    with st.expander("🤖 AI 设置", expanded=False):
        DEEPSEEK_API_KEY = st.text_input(
            "请输入 DeepSeek API Key（留空仅本地建议）", type="password"
        )
    analyze_btn = st.button("🚀 开始分析")

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
        if df is None or df.empty:
            st.error(f"代码 {code} 无可用行情数据！")
            st.stop()
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
            st.error(f"数据缺失: {miss}，实际字段: {df.columns.tolist()}")
            st.write(df.head())
            st.stop()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"行情数据接口异常: {e}")
        st.stop()

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

# ========== 技术指标 ==========
def add_indicators(df: pd.DataFrame, indicator: str):
    df["MA5"] = ta.sma(df["close"], length=5)
    df["MA20"] = ta.sma(df["close"], length=20)
    df["MA30"] = ta.sma(df["close"], length=30)

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
    if "MA30" in show_ma:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA30"], name="MA30"), row=1, col=1)
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

# ========== AI 止盈止损分析 ==========
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
def deepseek_probability_predict(tech_summary, fund_flow, news_list, api_key,
                                 hold_amount, hold_cost, latest_close, stop_profit, stop_loss):
    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "无相关新闻"
    flow_text = "\n".join([
        f"{d.get('日期', '')} 主力净流入: {format_money(d.get('主力净流入', d.get('ETF份额','无')))}"
        for d in fund_flow if "主力净流入" in d or "ETF份额" in d
    ])
    try:
        cost = float(hold_cost)
        amt = float(hold_amount)
        close = float(latest_close)
        profit = (close - cost) * amt if amt > 0 and cost > 0 else 0
        profit_rate = (close - cost) / cost * 100 if amt > 0 and cost > 0 else 0
        pos_desc = f"当前持有：{amt:.0f} 股，成本价：{cost:.2f}，现价：{close:.2f}，浮动盈亏：{profit:.2f} 元，盈亏率：{profit_rate:.2f}%"
        stop_line_desc = f"预设止盈线：{stop_profit:.2f}%，止损线：{stop_loss:.2f}%。"
        risk_flag = ""
        if profit_rate >= stop_profit:
            risk_flag = "【警告：已达到止盈线！建议考虑止盈卖出。】"
        elif profit_rate <= stop_loss:
            risk_flag = "【警告：已触及止损线！建议考虑止损离场。】"
    except:
        pos_desc = "当前未持有或成本/数量填写异常"
        stop_line_desc = ""
        risk_flag = ""

    prompt = f"""
以下是某只股票/ETF的全维度数据，请结合“技术面、资金流向、消息面、持仓盈亏、止盈止损线”进行AI分析。
分析内容：
1. 给出未来3日的上涨概率（%）、震荡概率（%）、下跌概率（%）；
2. 给出未来20日和30日的上涨/震荡/下跌概率，并结合 MA20 与 MA30 趋势关系说明原因；
3. 明确【买入/加仓/减仓/止盈/止损/观望】等操作建议，并详细说明原因（务必优先保障风控）。

【持仓信息】  
{pos_desc}
{stop_line_desc}
{risk_flag}

【技术面】  
{tech_summary}

【资金流向】  
{flow_text if flow_text else "暂无资金流数据"}

【消息面】  
{news_text}
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat","messages": [{"role": "user", "content": prompt}],"max_tokens": 600,"temperature": 0.5}
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek 概率预测出错: {e}"

# ========== 主程序 ==========
if analyze_btn:
    with st.spinner("数据加载中..."):
        df = fetch_realtime_kline(code, code_type)
        if df is None or df.empty:
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
        summary = f"收盘价:{latest['close']:.2f}, MA5:{latest['MA5']:.2f}, MA20:{latest['MA20']:.2f}, MA30:{latest['MA30']:.2f}"
        if indicator == "MACD":
            summary += f", MACD:{latest['MACD']:.3f}, 信号线:{latest['MACDs']:.3f}"
        st.subheader("📌 技术指标总结")
        st.write(summary)

        # 本地 MA20 vs MA30 趋势判断
        if latest["MA20"] > latest["MA30"]:
            st.info("【本地趋势判断】MA20 > MA30，中期趋势偏强。")
        else:
            st.info("【本地趋势判断】MA20 < MA30，中期趋势偏弱。")

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
            with st.spinner("DeepSeek AI 概率预测中..."):
                ai_text = deepseek_probability_predict(
                    summary, fund_flow, news_list, DEEPSEEK_API_KEY,
                    hold_amount, hold_cost, latest['close'], stop_profit, stop_loss
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
