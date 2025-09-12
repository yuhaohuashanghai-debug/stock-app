import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai

# ✅ 设置 OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="📊 A股策略分析", layout="wide")
st.title("📈 基于 AkShare + ChatGPT 的 A股技术分析与趋势预测")

# ✅ 获取行情数据函数
@st.cache_data(ttl=3600)
def fetch_data(code: str, start_date="20220101"):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        return df
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return pd.DataFrame()

# ✅ 技术指标计算
def add_indicators(df):
    df["MA5"] = ta.sma(df["close"], length=5)
    df["MA10"] = ta.sma(df["close"], length=10)
    df["MA20"] = ta.sma(df["close"], length=20)
    macd = ta.macd(df["close"])
    df["MACD"], df["MACD_H"], df["MACD_S"] = macd["MACD_12_26_9"], macd["MACDh_12_26_9"], macd["MACDs_12_26_9"]
    df["RSI"] = ta.rsi(df["close"], length=14)
    boll = ta.bbands(df["close"], length=20, std=2)
    df["BOLL_UP"], df["BOLL_MID"], df["BOLL_LOW"] = boll["BBU_20_2.0"], boll["BBM_20_2.0"], boll["BBL_20_2.0"]
    return df

# ✅ 趋势预测逻辑
def predict_trend(df):
    latest = df.iloc[-1]
    signals = []
    if latest["MACD"] > latest["MACD_S"]:
        signals.append("MACD 金叉 → 看涨")
    else:
        signals.append("MACD 死叉 → 看跌")

    if latest["RSI"] < 30:
        signals.append("RSI < 30 → 超卖反弹概率大")
    elif latest["RSI"] > 70:
        signals.append("RSI > 70 → 超买回落概率大")

    if latest["close"] > latest["BOLL_UP"]:
        signals.append("股价突破布林上轨 → 短期或回调")
    elif latest["close"] < latest["BOLL_LOW"]:
        signals.append("股价跌破布林下轨 → 可能反弹")

    return signals

# ✅ ChatGPT 解读模块
def ai_analysis(code, df, signals):
    latest = df.iloc[-1]
    prompt = f"""
你是一名专业的A股分析师，请根据以下数据写一份简短的研报风格解读，内容包含：技术面分析、风险提示、未来一周走势判断。
股票代码: {code}
日期: {latest['date'].strftime('%Y-%m-%d')}
收盘价: {latest['close']}
MA5: {latest['MA5']:.2f}, MA10: {latest['MA10']:.2f}, MA20: {latest['MA20']:.2f}
MACD: {latest['MACD']:.2f}, Signal: {latest['MACD_S']:.2f}
RSI: {latest['RSI']:.2f}
BOLL: 上轨 {latest['BOLL_UP']:.2f}, 中轨 {latest['BOLL_MID']:.2f}, 下轨 {latest['BOLL_LOW']:.2f}
信号总结: {"; ".join(signals)}
要求：语言专业、简洁，面向投资者，不要超过300字。
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "你是专业的证券分析师。"},
                      {"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.6
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ ChatGPT 分析失败: {e}"

# ✅ 页面交互
code = st.text_input("请输入6位股票代码", value="000001")
if st.button("分析股票"):
    df = fetch_data(code)
    if not df.empty:
        df = add_indicators(df)

        # 绘图
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5,0.25,0.25])
        fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
                                     low=df["low"], close=df["close"], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], name="MA5", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA10"], name="MA10", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_UP"], name="BOLL_UP", line=dict(width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_MID"], name="BOLL_MID", line=dict(width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_LOW"], name="BOLL_LOW", line=dict(width=1, dash="dot")), row=1, col=1)

        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量"), row=2, col=1)
        fig.add_trace(go.Bar(x=df["date"], y=df["MACD_H"], name="MACD柱状"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_S"], name="信号线"), row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # 趋势预测
        st.subheader("📌 技术信号解读")
        signals = predict_trend(df)
        for s in signals:
            st.write("- " + s)

        # AI 文字报告
        st.subheader("📝 ChatGPT 投资解读")
        report = ai_analysis(code, df, signals)
        st.write(report)
