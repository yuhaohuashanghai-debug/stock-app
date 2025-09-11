
import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak

openai.api_key = st.secrets["openai"]["api_key"]

st.set_page_config(page_title="AkShare + ChatGPT 股票分析", layout="wide")
st.title("📈 AkShare + ChatGPT 技术面股票分析")

def fetch_ak_kline(code):
    if len(code) != 6:
        st.error("股票代码应为6位数字，例如 000001 或 600519")
        return pd.DataFrame()
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20220101", adjust="qfq")
        df.rename(columns={"日期": "date", "收盘": "close"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        st.error(f"❌ AkShare 获取数据失败：{e}")
        return pd.DataFrame()

def analyze_tech(df):
    if 'close' not in df.columns or df['close'].isna().all():
        st.error("❌ 技术指标计算失败：未找到有效的收盘价数据")
        return df
    if len(df['close'].dropna()) < 30:
        st.warning("⚠️ 无法计算 MACD：数据量过少（至少需30条有效收盘价）")
        return df
    try:
        macd_df = ta.macd(df['close'])
        if macd_df is not None and not macd_df.empty:
            df = pd.concat([df, macd_df], axis=1)
            df.rename(columns={
                'MACD_12_26_9': 'MACD',
                'MACDs_12_26_9': 'MACD_signal',
                'MACDh_12_26_9': 'MACD_hist'
            }, inplace=True)
        else:
            st.warning("⚠️ 无法计算 MACD")
    except Exception as e:
        st.error(f"❌ MACD 计算异常：{e}")
    try:
        rsi_series = ta.rsi(df['close'])
        if rsi_series is not None:
            df['RSI'] = rsi_series
    except Exception as e:
        st.error(f"❌ RSI 计算异常：{e}")
    return df

from openai import OpenAI

client = OpenAI(api_key=st.secrets["openai"]["api_key"])  # ✅ 新写法初始化 client

def explain_by_gpt(stock_code, last_row):
    prompt = f"""
    请你分析股票 {stock_code}：
    当前价格：{last_row['close']:.2f}
    MACD值：{last_row['MACD']:.3f}, 信号线：{last_row['MACD_signal']:.3f}, 柱值：{last_row['MACD_hist']:.3f}
    RSI：{last_row['RSI']:.2f}
    请判断是否有买入/卖出/观望信号，并说明理由。
    """

    response = client.chat.completions.create(
        model="GPT‑4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

stock_code = st.text_input("请输入股票代码（6位，不带 SH/SZ 后缀）如 600519:")

if stock_code:
    with st.spinner("正在获取数据和分析中..."):
        df = fetch_ak_kline(stock_code)
        if df.empty:
            st.stop()
        df = analyze_tech(df)
        last_row = df.iloc[-1]

        st.subheader("📊 最近行情与技术指标")
        st.dataframe(df.tail(5)[['date', 'close', 'MACD', 'MACD_signal', 'RSI']].set_index('date'))

        st.subheader("🧠 ChatGPT 策略建议")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)
else:
    st.info("请输入6位股票代码，例如 000001 或 600519")
