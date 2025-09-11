
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

from openai import OpenAI
import os
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from openai import OpenAI
import os
import time

# 初始化 OpenAI 客户端（确保环境变量 OPENAI_API_KEY 已设置）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_by_gpt(stock_code, last_row):
    # 构造 prompt
    prompt = f"""
    请你作为一名资深投资分析师，分析股票「{stock_code}」的最新技术面情况。
    当前价格：{last_row.get('close', 'N/A')}
    MACD：{last_row.get('MACD', 'N/A')}，信号线：{last_row.get('MACD_signal', 'N/A')}
    RSI：{last_row.get('RSI', 'N/A')}
    
    请判断当前是否适合买入、卖出或观望，并用简洁语言给出理由（包括技术指标支撑）。
    """

    for _ in range(3):  # 最多重试3次
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # 推荐使用最新模型
                messages=[
                    {"role": "system", "content": "你是一个资深A股技术分析师"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)
            time.sleep(2)  # 简单重试
    return f"❌ GPT 分析失败：{err_msg}"

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
