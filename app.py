import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ========== 页面配置 ==========
st.set_page_config(page_title="📈 实时股票AI分析平台", layout="wide")
st.title("📊 实时股票技术分析 + 资金流向 + 消息面 + AI 趋势概率预测")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ========== 左侧控制面板 ==========
with st.sidebar:
    st.header("⚙️ 控制面板")

    with st.expander("📌 基础设置", expanded=True):
        code = st.text_input("股票代码（如 600519）", "600519")
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

# ========== 数据获取函数 ==========
@st.cache_data(ttl=300)
def fetch_kline(code: str, period="daily", start_date="20240101"):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period=period, start_date=start_date, adjust="qfq")
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_stock_news(code: str):
    try:
        df = ak.stock_news_em(symbol=code)
        for col in ["title", "新闻标题", "标题"]:
            if col in df.columns:
                return df[col].head(5).tolist()
        return ["未找到新闻标题字段"]
    except Exception as e:
        return [f"新闻获取失败: {e}"]

@st.cache_data(ttl=300)
def fetch_fund_flow(code: str):
    try:
        df = ak.stock_individual_fund_flow(stock=code)
        col_name = next((c for c in df.columns if "主力净流入" in c), None)
        if col_name:
            return df.tail(5)[["日期", col_name]].rename(columns={col_name: "主力净流入"}).to_dict("records")
        return []
    except Exception as e:
        return [{"error": str(e)}]

# ========== 技术指标 ==========
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

# ========== 图表绘制 ==========
def plot_chart(df: pd.DataFrame, code: str, indicator: str, show_ma: list, show_volume: bool):
    # 三层：K线、成交量、指标
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.2, 0.3],
                        vertical_spacing=0.05,
                        subplot_titles=(f"{code} K线图", "成交量", indicator))

    # K线
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线图"
    ), row=1, col=1)

    # 均线
    if "MA5" in show_ma:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], name="MA5"), row=1, col=1)
    if "MA20" in show_ma:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20"), row=1, col=1)

    # 成交量
    if show_volume:
        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量", opacity=0.4), row=2, col=1)

    # 指标
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

    fig.update_layout(height=1000, xaxis_rangeslider_visible=False, showlegend=True)
    return fig

# ========== AI 概率预测 ==========
def deepseek_probability_predict(tech_summary: str, fund_flow: list, news_list: list, api_key: str):
    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "无相关新闻"
    flow_text = "\n".join([f"{d['日期']} 主力净流入: {d['主力净流入']}" for d in fund_flow if "主力净流入" in d])

    prompt = f"""
以下是某只股票的多维度数据，请结合日线趋势、资金流向、技术指标和新闻，给出未来3日内的趋势概率预测：
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
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.5
    }

    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek 概率预测出错: {e}"

# ========== 主程序 ==========
if analyze_btn:
    df = fetch_kline(code, period="daily")
    if not df.empty:
        df = add_indicators(df, indicator)

        # 绘图
        st.plotly_chart(plot_chart(df, code, indicator, show_ma, show_volume), use_container_width=True)

        # 技术指标总结
        latest = df.iloc[-1]
        summary = f"收盘价:{latest['close']:.2f}, MA5:{latest['MA5']:.2f}, MA20:{latest['MA20']:.2f}"
        if indicator == "MACD":
            summary += f", MACD:{latest['MACD']:.3f}, 信号线:{latest['MACDs']:.3f}"
        st.subheader("📌 技术指标总结")
        st.write(summary)

        # 新闻
        news_list = fetch_stock_news(code)
        st.subheader("📰 实时消息面")
        for n in news_list:
            st.write("- " + n)

        # 资金流向
        fund_flow = fetch_fund_flow(code)
        st.subheader("💰 资金流向（近5日）")
        for f in fund_flow:
            if "主力净流入" in f:
                st.write(f"{f['日期']} 主力净流入: {f['主力净流入']}")

        # AI or 本地点评
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
