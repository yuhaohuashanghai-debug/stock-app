import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ========== 页面配置 ==========
st.set_page_config(page_title="📈 实时股票分析平台", layout="wide")
st.title("📊 实时股票技术分析 + 消息面 + DeepSeek AI 趋势预测")

# ========== DeepSeek API ==========
# 允许 secrets.toml 或网页输入
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", None)
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = st.text_input("请输入 DeepSeek API Key", type="password")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def deepseek_commentary(tech_summary: str, news_list: list, api_key: str):
    """调用 DeepSeek，结合技术面 + 消息面分析"""
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
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek 分析出错: {e}"

# ========== 调取行情数据 ==========
@st.cache_data(ttl=300)
def fetch_realtime_kline(code: str):
    """
    使用新浪财经接口获取日K数据（海外环境更稳定）
    code: 股票代码，例如 "600519" 或 "000001"
    """
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

# ========== 获取新闻 ==========
@st.cache_data(ttl=300)
def fetch_stock_news(code: str):
    try:
        df = ak.stock_news_em(symbol=code)  # 东方财富个股新闻
        return df["title"].head(5).tolist()
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

    # K线图
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], line=dict(width=1.5), name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], line=dict(width=1.5), name="MA20"), row=1, col=1)

    # 成交量
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量"), row=2, col=1)

    # MACD
    fig.add_trace(go.Bar(x=df["date"], y=df["MACDh"], name="MACDh"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], line=dict(width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], line=dict(width=1.2), name="信号线"), row=3, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False, showlegend=True)
    return fig

# ========== 主程序 ==========
code = st.text_input("请输入股票代码（如 600519）", "600519")

if st.button("分析"):
    if not DEEPSEEK_API_KEY:
        st.error("请先输入 DeepSeek API Key")
    else:
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

        # AI 综合分析
        with st.spinner("DeepSeek AI 综合分析中..."):
            ai_text = deepseek_commentary(summary, news_list, DEEPSEEK_API_KEY)
            st.subheader("🤖 AI 综合分析与趋势预测")
            st.write(ai_text)
