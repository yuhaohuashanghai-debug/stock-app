import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import jqdatasdk

# ---------------------- 🔐 密钥配置 ----------------------
openai.api_key = st.secrets["openai_api_key"]
jq_user = st.secrets["joinquant"]["username"]
jq_pass = st.secrets["joinquant"]["password"]

# 登录 JoinQuant
try:
    jqdatasdk.auth(jq_user, jq_pass)
    st.success("✅ 聚宽登录成功")
except Exception as e:
    st.error(f"❌ 聚宽登录失败: {e}")
    st.stop()

# ---------------------- 📊 页面配置 ----------------------
st.set_page_config(page_title="智能股票分析助手", layout="wide")
st.title("📈 ChatGPT + 技术面 股票分析工具")

stock_code = st.text_input("请输入股票代码 (例如: 000001.SZ or 600519.SH):")

# ---------------------- 🔌 数据源接入 (聚宽) ----------------------
def fetch_kline_from_jq(stock_code):
    try:
        if '.' not in stock_code:
            stock_code = stock_code + '.XSHE' if stock_code.startswith('0') else stock_code + '.XSHG'

        df = jqdatasdk.get_price(stock_code, start_date="2023-01-01", end_date="2025-12-31", frequency="daily", fields=["open", "close", "high", "low", "volume"])

        if df is None or df.empty:
            st.warning("⚠️ 获取K线数据为空，请检查股票代码或权限")
            return pd.DataFrame()

        df = df.reset_index().rename(columns={"index": "date"})
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"❌ 获取行情数据失败（聚宽）：{e}")
        return pd.DataFrame()

# ---------------------- 📈 技术面分析 ----------------------
def analyze_tech(df):
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.macd(df['close'])
    df['RSI'] = ta.rsi(df['close'])
    df['MACD'] = df['MACD'].astype(float)
    df['MACD_signal'] = df['MACD_signal'].astype(float)
    df['MACD_hist'] = df['MACD_hist'].astype(float)
    df['RSI'] = df['RSI'].astype(float)
    return df

# ---------------------- 🤖 ChatGPT 解释 ----------------------
def explain_by_gpt(stock_code, last_row):
    prompt = f"""
    请你分析股票 {stock_code}：
    当前价格：{float(last_row['close']):.2f}
    MACD值：{float(last_row['MACD']):.3f}, 信号线：{float(last_row['MACD_signal']):.3f}, 柱值：{float(last_row['MACD_hist']):.3f}
    RSI：{float(last_row['RSI']):.2f}
    请判断是否有买入/卖出/观望信号，并说明理由。
    """
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ---------------------- 🔄 主程序逻辑 ----------------------
if stock_code:
    with st.spinner("正在获取数据和分析中..."):
        try:
            df = fetch_kline_from_jq(stock_code)
            if df.empty:
                st.stop()
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
