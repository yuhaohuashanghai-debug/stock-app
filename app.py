
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
import json
import plotly.graph_objects as go
import plotly.express as px

if USE_STREAMLIT:
    openai.api_key = st.secrets["general"]["openai_api_key"]
else:
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

if USE_STREAMLIT:
    st.set_page_config(page_title="智能股票分析助手", layout="wide")
    st.title("📈 ChatGPT + 技术面 股票分析工具")
    stock_code = st.text_input("请输入股票代码（如: 600519 或 000001）", value="600519")
    show_candle = st.checkbox("显示K线图", value=True)
    show_macd = st.checkbox("显示MACD图", value=True)
    show_rsi = st.checkbox("显示RSI图", value=False)
else:
    stock_code = input("请输入股票代码（如: 600519）：")
    show_candle, show_macd, show_rsi = True, True, False

# 新浪财经接口封装（带 JSONP 修复）
def fetch_kline_from_sina(stock_code):
    market = "sh" if stock_code.startswith("6") else "sz"
    symbol = f"{market}{stock_code}"
    url = f"https://finance.sina.com.cn/realstock/company/{symbol}/hisdata/klc_kl.js"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise ValueError("请求新浪财经失败")
        text = response.text

        json_str = text[text.find("["):text.rfind("]")+1]
        raw_data = json.loads(json_str)
        df = pd.DataFrame(raw_data)
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        for col in ["open", "close", "high", "low"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        msg = f"❌ 获取行情数据失败（新浪财经）：{e}"
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
    df = fetch_kline_from_sina(stock_code)
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
