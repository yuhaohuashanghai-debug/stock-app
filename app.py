import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ========== 页面配置 ==========
st.set_page_config(page_title="📈 股票AI分析平台", layout="wide")
st.title("📊 实时股票分析 + DeepSeek AI 趋势预测")

# ========== DeepSeek API ==========
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def deepseek_commentary(tech_summary: str):
    """调用 DeepSeek 生成分析点评和趋势预测"""
    prompt = f"""
以下是某只股票的实时技术指标，请结合 MACD、均线、成交量，先点评行情，
再预测短期趋势（上涨/震荡/下跌），理由要清晰：

{tech_summary}
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek 分析出错: {e}"

# ========== 调取实时数据 ==========
@st.cache_data(ttl=300)  # 缓存5分钟
def fetch_realtime_kline(code: str):
    """
    直接调取新浪财经的实时日K数据
    code: 股票代码, 例如 "600519"
    """
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
    df.rename(columns={"日期":"date", "开盘":"open", "收盘":"close",
                       "最高":"high", "最低":"low", "成交量":"volume"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df

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
    df = fetch_realtime_kline(code)
    df = add_indicators(df)

    st.plotly_chart(plot_chart(df, code), use_container_width=True)

    # 最新指标
    latest = df.iloc[-1]
    summary = f"""
收盘价: {latest['close']:.2f}, 
MA5: {latest['MA5']:.2f}, 
MA20: {latest['MA20']:.2f}, 
MACD: {latest['MACD']:.3f}, 
信号线: {latest['MACDs']:.3f}, 
成交量: {latest['volume']}
"""
    st.subheader("📌 实时技术指标总结")
    st.write(summary)

    # AI 分析 + 趋势预测
    with st.spinner("DeepSeek AI 正在分析..."):
        ai_text = deepseek_commentary(summary)
        st.subheader("🤖 AI 分析与趋势预测")
        st.write(ai_text)
