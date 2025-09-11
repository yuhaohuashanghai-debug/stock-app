
import streamlit as st
from akshare import stock_zh_a_hist
import pandas as pd
import matplotlib.pyplot as plt
import talib
from explain import explain_by_gpt

st.set_page_config(page_title="ChatGPT + 技术面 股票分析工具", layout="wide")

st.title("📈 ChatGPT + 技术面 股票分析工具")

stock_code = st.text_input("请输入股票代码（如: 000001.SZ or 600519.SH):", value="000001.SZ")

show_k = st.checkbox("显示K线图", value=True)
show_macd = st.checkbox("显示MACD图", value=True)
show_rsi = st.checkbox("显示RSI图", value=True)

if stock_code:
    with st.spinner("正在获取数据和分析中..."):
        try:
            df = stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20220101", end_date="20500101")
            df.rename(columns={"日期": "date", "收盘": "close"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.set_index("date", inplace=True)

            # 计算技术指标
            df["MACD"], df["MACD_signal"], _ = talib.MACD(df["close"])
            df["RSI"] = talib.RSI(df["close"])

            st.success("✅ 数据获取成功")

            if show_k:
                st.subheader("K线图")
                fig, ax = plt.subplots()
                df["close"].plot(ax=ax, title="收盘价")
                st.pyplot(fig)

            if show_macd:
                st.subheader("MACD")
                fig, ax = plt.subplots()
                df[["MACD", "MACD_signal"]].plot(ax=ax)
                st.pyplot(fig)

            if show_rsi:
                st.subheader("RSI")
                fig, ax = plt.subplots()
                df["RSI"].plot(ax=ax)
                st.pyplot(fig)

            # 分析建议
            st.subheader("🧠 ChatGPT 策略建议")
            last_row = df.iloc[-1].to_dict()
            suggestion = explain_by_gpt(stock_code, last_row)
            st.success(suggestion)

        except Exception as e:
            st.error(f"❌ 获取行情数据失败：{e}")
