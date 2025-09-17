import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# ========== 页面配置 ==========
st.set_page_config(page_title="📈 实时股票分析平台", layout="wide")
st.title("📊 实时股票技术分析 + 消息面 + AI 趋势预测")

# ========== API Key 输入 ==========
DEEPSEEK_API_KEY = st.text_input("请输入 DeepSeek API Key（留空则只做本地技术点评）", type="password")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ========== DeepSeek 请求函数 ==========
def deepseek_commentary(tech_summary: str, news_list: list, api_key: str):
    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "无相关新闻"
    prompt = f"""
以下是某只股票的最新情况，请结合技术指标与消息面综合点评，并预测短期趋势（上涨/震荡/下跌）：

【技术面】
{tech_summary}

【消息面】
{news_text}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.7
    }

    # 配置重试机制
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        resp = session.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek 分析出错: {e}"

# ========== 行情数据（日K） ==========
@st.cache_data(ttl=300)
def fetch_realtime_kline(code: str):
    if code.startswith("6"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"

    df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    df = df.reset_index()
    df.rename(columns={
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume"
    }, inplace=True)
    return df

# ========== 新闻接口 ==========
@st.cache_data(ttl=300)
def fetch_stock_news(code: str):
    try:
        df = ak.stock_news_em(symbol=code)
        # 自动识别标题字段
        for col in ["title", "新闻标题", "标题"]:
            if col in df.columns:
                return df[col].head(5).tolist()
        return ["未找到新闻标题字段"]
    except Exception as e:
        return [f"新闻获取失败: {e}"]

# ========== 技术指标 ==========
def add_indicators(df: pd.DataFrame):
    df["MA5"] = ta.sma(df["close"], length=5)
    df["MA20"] = ta.sma(df["close"], length=20)
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACDh"] = macd["MACDh_12_26_9"]
    df["MACDs"] = macd["MACDs_12_26_9"]
    return df

# ========== 绘制图表 ==========
def plot_chart(df: pd.DataFrame, code: str):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.2, 0.3],
                        vertical_spacing=0.05,
                        subplot_titles=(f"{code} K线图", "成交量", "MACD"))

    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], line=dict(width=1.5), name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], line=dict(width=1.5), name="MA20"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量"), row=2, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["MACDh"], name="MACDh"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], line=dict(width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], line=dict(width=1.2), name="信号线"), row=3, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False, showlegend=True)
    return fig

# ========== 主程序 ==========
code = st.text_input("请输入股票代码（如 600519）", "600519")

if st.button("分析"):
    df = fetch_realtime_kline(code)
    df = add_indicators(df)
    st.plotly_chart(plot_chart(df, code), use_container_width=True)

    # 技术指标总结
    latest = df.iloc[-1]
    summary = f"收盘价:{latest['close']:.2f}, MA5:{latest['MA5']:.2f}, MA20:{latest['MA20']:.2f}, MACD:{latest['MACD']:.3f}, 信号线:{latest['MACDs']:.3f}, 成交量:{latest['volume']}"
    st.subheader("📌 技术指标总结")
    st.write(summary)

    # 新闻
    news_list = fetch_stock_news(code)
    st.subheader("📰 实时消息面")
    for n in news_list:
        st.write("- " + n)

    # AI 分析 or 本地点评
    if DEEPSEEK_API_KEY:
        with st.spinner("DeepSeek AI 综合分析中..."):
            ai_text = deepseek_commentary(summary, news_list, DEEPSEEK_API_KEY)
            st.subheader("🤖 AI 综合分析与趋势预测")
            st.write(ai_text)
    else:
        st.subheader("🤖 本地技术面点评")
        if latest["MACD"] > latest["MACDs"]:
            st.write("MACD 金叉，短期有反弹可能。")
        elif latest["MACD"] < latest["MACDs"]:
            st.write("MACD 死叉，短期下行动能较大。")
        else:
            st.write("MACD 持平，市场观望情绪浓。")
