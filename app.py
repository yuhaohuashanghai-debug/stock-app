import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ✅ 页面初始化
st.set_page_config(page_title="突破型短线趋势策略", layout="wide")
st.title("📈 突破型短线趋势策略分析 (AkShare + Streamlit)")

# ✅ 股票代码输入
code = st.text_input("请输入股票或ETF代码 (6位):", "159995")

# ✅ 参数滑块
period_high = st.slider("平台突破周期（日）", 10, 60, 20)
volume_ratio_threshold = st.slider("放量阈值 (倍数)", 1.0, 3.0, 1.5, 0.1)
max_drawdown = st.slider("最大回撤止损%", 3, 15, 7)
target_profit = st.slider("目标止盈%", 5, 20, 12)

# ✅ 数据获取
@st.cache_data(ttl=3600)
def fetch_kline(symbol):
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20220101", adjust="qfq")
        df.rename(columns={"日期": "date", "收盘": "close", "开盘": "open", "最高": "high",
                          "最低": "low", "成交量": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)
        return df
    except:
        return pd.DataFrame()

df = fetch_kline(code)
if df.empty:
    st.warning("数据获取失败，请检查股票代码是否正确。")
    st.stop()

# ✅ 指标计算
df["MA5"] = df["close"].rolling(5).mean()
df.ta.macd(close="close", append=True)
df.ta.rsi(length=14, append=True)
df["vol_ma10"] = df["volume"].rolling(10).mean()

# ✅ 信号生成逻辑
signal = []
holding = False
entry_price = 0
for i in range(period_high, len(df)):
    today = df.iloc[i]
    prev = df.iloc[i-1]
    price = today["close"]
    vol = today["volume"]
    high_platform = df["close"].iloc[i-period_high:i].max()
    vol_ma = today["vol_ma10"]
    is_breakout = price > high_platform
    is_volume = vol > vol_ma * volume_ratio_threshold
    is_macd = today["MACD_12_26_9"] > 0 and prev["MACD_12_26_9"] <= 0
    is_rsi = today["RSI_14"] > 55
    
    if not holding and is_breakout and is_volume and (is_macd or is_rsi):
        holding = True
        entry_price = price
        signal.append("buy")
    elif holding:
        drawdown = (price - entry_price) / entry_price * 100
        if drawdown <= -max_drawdown or drawdown >= target_profit or price < prev["close"]:
            holding = False
            signal.append("sell")
        else:
            signal.append("hold")
    else:
        signal.append("none")

df = df.iloc[period_high:].copy()
df["signal"] = signal

# ✅ 可视化图表
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.05,
                    subplot_titles=("K线图", "成交量", "MACD & RSI"))

# K线图
fig.add_trace(go.Candlestick(x=df.index,
                             open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                             name="K线"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], mode="lines", name="MA5", line=dict(width=1)), row=1, col=1)

# 信号标注
buy_signals = df[df["signal"] == "buy"]
sell_signals = df[df["signal"] == "sell"]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["close"],
                         mode="markers", marker=dict(symbol="triangle-up", color="green", size=10),
                         name="买入信号"), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["close"],
                         mode="markers", marker=dict(symbol="triangle-down", color="red", size=10),
                         name="卖出信号"), row=1, col=1)

# 成交量
fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="成交量"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["vol_ma10"], name="均量10日", line=dict(width=1, dash="dot")), row=2, col=1)

# MACD & RSI
fig.add_trace(go.Scatter(x=df.index, y=df["MACD_12_26_9"], name="MACD", line=dict(color="orange")), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI", line=dict(color="purple")), row=3, col=1)

fig.update_layout(height=900, width=1200, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ✅ 信号结果表
st.subheader("📌 交易信号明细")
st.dataframe(df[df["signal"].isin(["buy", "sell"])]
             [["close", "volume", "MACD_12_26_9", "RSI_14", "signal"]]
             .rename(columns={"close": "收盘价", "volume": "成交量", "signal": "信号"}))
