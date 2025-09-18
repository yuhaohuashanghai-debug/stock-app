import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests

st.set_page_config(page_title="📈 实时股票&ETF AI分析平台", layout="wide")
st.title("📊 实时股票/ETF 技术分析 + 资金流向 + 消息面 + AI 趋势概率预测")

# ========== 控制面板 ==========
with st.sidebar:
    st.header("⚙️ 控制面板")
    with st.expander("📌 基础设置", expanded=True):
        code_type = st.radio("类型", ["A股", "ETF"], horizontal=True)
        code = st.text_input("股票/ETF代码（如 600519 或 510300）", "600519")
        show_volume = st.checkbox("显示成交量", value=True)
    with st.expander("📊 指标设置", expanded=True):
        show_ma = st.multiselect("显示均线", ["MA5", "MA20"], default=["MA5", "MA20"])
        indicator = st.selectbox("选择额外指标", ["MACD", "RSI", "BOLL", "KDJ"])
    with st.expander("🤖 AI 设置", expanded=False):
        DEEPSEEK_API_KEY = st.text_input(
            "请输入 DeepSeek API Key（留空则只做本地技术点评）",
            type="password"
        )
    analyze_btn = st.button("🚀 开始分析")

# ========== 核心：数据获取通用接口 ==========
@st.cache_data(ttl=300)
def fetch_realtime_kline(code: str, code_type: str):
    import akshare as ak
    import pandas as pd

    try:
        if code_type == "A股":
            symbol = f"sh{code}" if code.startswith("6") else f"sz{code}"
            try:
                df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
            except Exception as e:
                st.error(f"A股接口报错：{e}")
                return pd.DataFrame()
        else:
            # ETF多接口自动兜底
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
            st.error(f"代码 {code} 无可用行情数据（所有接口返回空）！请换ETF或股票代码再试。")
            st.stop()
        # 字段自动映射
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
            st.error(f"数据缺失关键字段: {miss}，实际字段: {df.columns.tolist()}")
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
            return ["ETF暂无个股新闻，建议关注指数、主题或市场消息"]
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
            # ETF：用成交额/成交量（资金流近似），ak.fund_etf_hist_em最稳
            df = ak.fund_etf_hist_em(symbol=code)
            df = df.tail(5)
            if "日期" in df.columns and "成交额" in df.columns and "成交量" in df.columns:
                return df[["日期", "成交额", "成交量"]].to_dict("records")
            else:
                return [{"error": f"ETF接口无成交额/量字段，返回: {df.columns.tolist()}"}]
    except Exception as e:
        return [{"error": str(e)}]

@st.cache_data(ttl=300)
def fetch_stock_concepts(code: str, code_type: str):
    try:
        if code_type == "ETF":
            return ["ETF指数/主题型", "具体主题可参考ETF名称"]
        all_concepts = ak.stock_board_concept_name_ths()
        concept_col = None
        for col in ["名称", "板块名称", "name"]:
            if col in all_concepts.columns:
                concept_col = col
                break
        if not concept_col:
            return [f"未找到板块字段，现有字段: {all_concepts.columns.tolist()}"]
        result = []
        for name in all_concepts[concept_col]:
            try:
                cons = ak.stock_board_concept_cons_ths(symbol=name)
                for code_col in ["代码", "code"]:
                    if code_col in cons.columns and code in cons[code_col].astype(str).tolist():
                        result.append(name)
            except:
                continue
        return result if result else ["未找到所属概念板块"]
    except Exception as e:
        return [f"获取板块失败: {e}"]

@st.cache_data(ttl=300)
def fetch_concept_fund_flow(concept_name=None):
    import akshare as ak
    import pandas as pd
    try:
        # 新接口：用hist接口，取最近5日，含主力资金流/涨跌幅
        if concept_name is None:
            # 可选：返回所有板块最新行情/或遍历所有主流板块
            concept_list = ak.stock_board_concept_name_ths()
            flows = []
            for name in concept_list['name'].head(20):  # 只查前20个热门，避免API频繁
                try:
                    df = ak.stock_board_concept_hist_ths(symbol=name, start_date="20230901", end_date=pd.Timestamp.today().strftime('%Y%m%d'))
                    if not df.empty:
                        last = df.iloc[-1]
                        flows.append({
                            "板块名称": name,
                            "涨跌幅": last.get("涨跌幅", None),
                            "主力净流入": last.get("主力资金净流入", None)
                        })
                except:
                    continue
            return pd.DataFrame(flows)
        else:
            df = ak.stock_board_concept_hist_ths(symbol=concept_name, start_date="20230901", end_date=pd.Timestamp.today().strftime('%Y%m%d'))
            return df.tail(5)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

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

# ========== AI 概率预测 ==========
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
def deepseek_probability_predict(tech_summary: str, fund_flow: list, news_list: list, api_key: str):
    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "无相关新闻"
    flow_text = "\n".join([
        f"{d['日期']} 主力净流入: {format_money(d.get('主力净流入', d.get('ETF份额','无')))}"
        for d in fund_flow if "主力净流入" in d or "ETF份额" in d
    ])
    prompt = f"""
以下是某只股票/ETF的多维度数据，请结合日线趋势、资金流向、技术指标和新闻，给出未来3日内的趋势概率预测：
- 上涨概率（%）
- 震荡概率（%）
- 下跌概率（%）
并简要说明原因。

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 图表", "📰 新闻", "💰 资金流", "🤖 AI/本地分析", "📊 板块概念联动"]
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
            st.subheader("💰 ETF成交额/成交量（近5日，仅供资金流参考）")
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
        if DEEPSEEK_API_KEY:
            with st.spinner("DeepSeek AI 概率预测中..."):
                ai_text = deepseek_probability_predict(summary, fund_flow, news_list, DEEPSEEK_API_KEY)
                st.subheader("📊 AI 趋势概率预测")
                st.write(ai_text)
        else:
            st.subheader("🤖 本地技术面点评")
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

    with tab5:
        st.subheader("📊 板块概念联动分析")
        if code_type == "A股":
            try:
                all_concepts = ak.stock_board_concept_name_ths()
                st.write("🔍 概念板块接口返回字段:", all_concepts.columns.tolist())
                st.dataframe(all_concepts.head())
            except Exception as e:
                st.write("获取概念板块失败:", str(e))
            try:
                flow_df_raw = ak.stock_board_concept_fund_flow_ths()
                st.write("🔍 板块资金流接口返回字段:", flow_df_raw.columns.tolist())
                st.dataframe(flow_df_raw.head())
            except Exception as e:
                st.write("获取资金流失败:", str(e))
            concepts = fetch_stock_concepts(code, code_type)
            if concepts:
                st.write("所属概念板块:", "、".join(concepts))
                flow_df = fetch_concept_fund_flow()
                if not flow_df.empty and "error" not in flow_df.columns:
                    flow_df = flow_df[flow_df["板块名称"].isin(concepts)]
                    if not flow_df.empty:
                        flow_df["主力净流入数值"] = pd.to_numeric(flow_df["主力净流入"], errors="coerce")
                        flow_df["涨跌幅数值"] = pd.to_numeric(flow_df["涨跌幅"], errors="coerce")
                        flow_df = flow_df.sort_values("主力净流入数值", ascending=False)
                        st.dataframe(flow_df[["板块名称", "涨跌幅", "主力净流入"]])
                        heatmap_df = pd.melt(
                            flow_df,
                            id_vars=["板块名称"],
                            value_vars=["主力净流入数值", "涨跌幅数值"],
                            var_name="指标",
                            value_name="数值"
                        )
                        fig = px.imshow(
                            heatmap_df.pivot(index="指标", columns="板块名称", values="数值").values,
                            labels=dict(x="板块名称", y="指标", color="数值"),
                            x=flow_df["板块名称"].tolist(),
                            y=["主力净流入", "涨跌幅"],
                            color_continuous_scale="RdYlGn"
                        )
                        fig.update_layout(height=500, margin=dict(l=40, r=40, t=40, b=40))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("暂无板块资金流数据")
                else:
                    st.write("板块资金流获取失败")
            else:
                st.write("未找到相关概念板块")
        else:
            st.write("ETF主题/指数板块：", fetch_stock_concepts(code, code_type))
            st.write("ETF多为主题指数，无A股概念板块资金流联动。")

