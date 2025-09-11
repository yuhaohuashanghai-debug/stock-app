import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import requests

openai.api_key = st.secrets["general"]["openai_api_key"]

st.set_page_config(page_title="智能股票分析助手", layout="wide")
st.title("📈 ChatGPT + 技术面 股票分析工具")

stock_code = st.text_input("请输入股票代码 (例如: 000001.SZ or 600519.SH):")

def fetch_eastmoney_kline(stock_code):
    code = stock_code.replace(".SH", "1").replace(".SZ", "0")
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": code,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": 1,
        "beg": "20220101",
        "end": "20500101"
    }
    res = requests.get(url, params=params).json()
    klines = res['data']['klines']
    df = pd.DataFrame([x.split(',') for x in klines], columns=[
        'date','open','close','high','low','volume','turnover','amplitude','chg_pct','chg_amt','turnover_rate'])
    df['close'] = df['close'].astype(float)
    return df

def analyze_tech(df):
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.macd(df['close'])
    df['RSI'] = ta.rsi(df['close'])
    return df

def explain_by_gpt(stock_code, last_row):
    prompt = f"""
    请你分析股票 {stock_code}：
    当前价格：{last_row['close']:.2f}
    MACD值：{last_row['MACD']:.3f}, 信号线：{last_row['MACD_signal']:.3f}, 柱值：{last_row['MACD_hist']:.3f}
    RSI：{last_row['RSI']:.2f}
    请判断是否有买入/卖出/观望信号，并说明理由。
    """
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

if stock_code:
    with st.spinner("正在获取数据和分析中..."):
        try:
            df = fetch_eastmoney_kline(stock_code)
            df = analyze_tech(df)
            last_row = df.iloc[-1]

            st.subheader("📊 最近行情与技术指标")
            st.dataframe(df.tail(5)[['date', 'close', 'MACD', 'MACD_signal', 'RSI']].set_index('date'))

            st.subheader("🧠 ChatGPT 策略建议")
            suggestion = explain_by_gpt(stock_code, last_row)
            st.markdown(suggestion)

        except Exception as e:
            st.error(f"出错啦：{e}")
else:
    st.info("请输入股票代码，例如 000001.SZ 或 600519.SH")
