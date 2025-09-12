import streamlit as st
import pandas as pd
import akshare as ak
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# 页面初始化
st.set_page_config(page_title="短/中线趋势选股系统", layout="wide")
st.title("📈 趋势跟踪选股系统（A股 / ETF / AkShare）")

# 参数设置
days_high = st.sidebar.slider("突破周期（日高）", 10, 60, 20)
volume_ratio = st.sidebar.slider("放量阈值倍数", 1.0, 3.0, 1.5, 0.1)
macd_enable = st.sidebar.checkbox("启用MACD金叉确认", value=True)
rsi_enable = st.sidebar.checkbox("启用RSI强势过滤 (>55)", value=True)
start_date = st.sidebar.date_input("数据起始日期", value=datetime(2022, 1, 1))

# 股票列表输入
codes_input = st.text_area("输入股票代码列表（用逗号分隔）", "159995, 588170, 518880")
codes = [x.strip() for x in codes_input.split(',') if len(x.strip()) == 6]

# 数据获取
def get_data(code):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date.strftime("%Y%m%d"), adjust="qfq")
        df.rename(columns={"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low", "成交量": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df["MA5"] = df["close"].rolling(5).mean()
        df.ta.macd(close='close', append=True)
        df.ta.rsi(length=14, append=True)
        df["vol_ma10"] = df["volume"].rolling(10).mean()
        return df
    except:
        return pd.DataFrame()

# 策略逻辑
def check_signal(df):
    if len(df) < days_high + 5:
        return False
    today = df.iloc[-1]
    prev = df.iloc[-2]
    highest = df["close"].iloc[-days_high:].max()
    vol_check = today["volume"] > today["vol_ma10"] * volume_ratio
    price_check = today["close"] > highest
    macd_check = today["MACD_12_26_9"] > 0 and prev["MACD_12_26_9"] <= 0 if macd_enable else True
    rsi_check = today["RSI_14"] > 55 if rsi_enable else True
    return vol_check and price_check and macd_check and rsi_check

# 图形化函数
def plot_chart(df, code):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="K线"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], mode="lines", name="MA5", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_12_26_9"], mode="lines", name="MACD", yaxis="y2", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], mode="lines", name="RSI", yaxis="y3", line=dict(color="purple")))

    fig.update_layout(
        title=f"📊 {code} K线趋势图",
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(title="价格"),
        yaxis2=dict(title="MACD", overlaying="y", side="right", showgrid=False, position=0.95),
        yaxis3=dict(title="RSI", overlaying="y", side="right", showgrid=False, position=1.0),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# 筛选结果
report = []
for code in codes:
    df = get_data(code)
    if df.empty:
        continue
    if check_signal(df):
        today = df.iloc[-1]
        report.append({
            "代码": code,
            "收盘价": today["close"],
            "MA5": today["MA5"],
            "MACD": today["MACD_12_26_9"],
            "RSI": today["RSI_14"],
            "成交量": today["volume"],
            "信号": "✅ 突破 + 放量"
        })
        plot_chart(df, code)  # 显示图形

# 展示结果
st.subheader("📊 满足趋势选股条件的股票")
if report:
    st.success(f"共选出 {len(report)} 只股票/ETF 满足当前趋势策略")
    st.dataframe(pd.DataFrame(report))
else:
    st.warning("暂无符合条件的股票/ETF")
