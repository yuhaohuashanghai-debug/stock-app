
try:
    import streamlit as st
    USE_STREAMLIT = True
except ModuleNotFoundError:
    print("[INFO] Streamlit 未安装，进入非交互模式。")
    USE_STREAMLIT = False

import pandas as pd
import pandas_ta as ta
import openai
import requests
import os
import plotly.graph_objects as go
import plotly.express as px

if USE_STREAMLIT:
    openai.api_key = st.secrets["general"]["openai_api_key"]
else:
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

if USE_STREAMLIT:
    st.set_page_config(page_title="智能股票分析助手", layout="wide")
    st.title("📈 ChatGPT + 技术面 股票分析工具")
    stock_code = st.text_input("请输入股票代码（如: 000001.SZ）", value="600519.SH")
    show_candle = st.checkbox("显示K线图", value=True)
    show_macd = st.checkbox("显示MACD图", value=True)
    show_rsi = st.checkbox("显示RSI图", value=False)
else:
    stock_code = input("请输入股票代码（如: 000001.SZ）：")
    show_candle, show_macd, show_rsi = True, True, False

def fetch_kline_from_eastmoney(stock_code):
    if '.' not in stock_code:
        stock_code = stock_code + '.SH' if stock_code.startswith('6') else stock_code + '.SZ'
    secid = stock_code.replace(".SH", "1").replace(".SZ", "0")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": 1,
        "beg": "20220101",
        "end": "20500101"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code != 200:
            raise ValueError(f"请求失败，状态码: {response.status_code}")
        res = response.json()
        if not isinstance(res, dict) or "data" not in res or res["data"] is None:
            raise ValueError("返回数据格式不正确或为空")
        klines = res["data"].get("klines", [])
        if not klines:
            raise ValueError("未获取到任何K线数据")

        df = pd.DataFrame([row.split(",") for row in klines],
                          columns=["date", "open", "close", "high", "low", "volume", "turnover",
                                   "amplitude", "chg_pct", "chg_amt", "turnover_rate"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        msg = f"❌ 获取行情数据失败：{e}"
        if USE_STREAMLIT:
            st.error(msg)
        else:
            print(msg)
        return pd.DataFrame()

def analyze_tech(df):
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.macd(df['close'])
    df['RSI'] = ta.rsi(df['close'])
    return df

def explain_by_gpt(stock_code, last_row):
    prompt = f"""请你分析股票 {stock_code}：
当前价格：{last_row['close']:.2f}
MACD值：{last_row['MACD']:.3f}, 信号线：{last_row['MACD_signal']:.3f}, 柱值：{last_row['MACD_hist']:.3f}
RSI：{last_row['RSI']:.2f}
请判断是否有买入/卖出/观望信号，并说明理由。"""
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def plot_charts(df):
    plots = []
    if show_candle:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K线"))
        fig.update_layout(title="K线图", xaxis_rangeslider_visible=False)
        plots.append(fig)

    if show_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['date'], y=df['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], mode='lines', name='Signal'))
        fig_macd.add_trace(go.Bar(x=df['date'], y=df['MACD_hist'], name='Hist'))
        fig_macd.update_layout(title="MACD", height=300)
        plots.append(fig_macd)

    if show_rsi:
        fig_rsi = px.line(df, x='date', y='RSI', title="RSI")
        plots.append(fig_rsi)

    return plots

if stock_code:
    df = fetch_kline_from_eastmoney(stock_code)
    if not df.empty:
        df = analyze_tech(df)
        last_row = df.iloc[-1]

        if USE_STREAMLIT:
            st.subheader("📊 技术指标数据")
            st.dataframe(df.tail(5)[['date', 'close', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI']].set_index('date'))

            st.subheader("📈 图表展示")
            for fig in plot_charts(df):
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("🧠 ChatGPT 策略建议")
            suggestion = explain_by_gpt(stock_code, last_row)
            st.markdown(suggestion)
        else:
            print(df.tail())
            print(explain_by_gpt(stock_code, last_row))
    else:
        if USE_STREAMLIT:
            st.warning("未获取到任何数据")
        else:
            print("无数据")
