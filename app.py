import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
from jqdatasdk import auth, get_price
from datetime import datetime

# 设置 OpenAI API Key
openai.api_key = st.secrets["general"]["openai_api_key"]

# 聚宽登录
@st.cache_data(show_spinner="🔐 正在登录聚宽...")
def jq_login():
    try:
        auth(st.secrets["jq"]["account"], st.secrets["jq"]["password"])
        st.success("✅ 聚宽登录成功")
    except Exception as e:
        st.error(f"❌ 聚宽登录失败：{e}")
        st.stop()

# 获取K线数据（聚宽）
def fetch_kline_from_jq(stock_code):
    jq_login()

    if '.' not in stock_code:
        stock_code += '.XSHG' if stock_code.startswith('6') else '.XSHE'

    try:
        start_date = "2024-06-05"
        end_date = "2024-06-05"
        df = get_price(stock_code, start_date=start_date, end_date=end_date, frequency='daily')
        if df is None or df.empty:
            st.warning("⚠️ 聚宽返回空数据")
            return pd.DataFrame()
        df = df.rename(columns={
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
        })
        df = df.reset_index()
        return df
    except Exception as e:
        st.error(f"❌ 获取行情数据失败（聚宽）：{e}")
        return pd.DataFrame()

# 页面配置
st.set_page_config(page_title="智能股票分析助手", layout="wide")
st.title("📈 ChatGPT + 技术面 股票分析工具")

stock_code = st.text_input("请输入股票代码 (例如: 000001.SZ or 600519.SH):")

def analyze_tech(df):
    # 确保 'close' 列存在且不为空
    if 'close' not in df.columns or df['close'].isna().all():
        st.error("❌ 技术指标计算失败：未找到有效的收盘价数据")
        return df

    # 至少需要 30 条数据计算 MACD
    if len(df['close'].dropna()) < 30:
        st.warning("⚠️ 无法计算 MACD：数据量过少（至少需30条有效收盘价）")
        return df

    # 计算 MACD（添加异常处理）
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
            st.warning("⚠️ 无法计算 MACD 指标，macd_df 为空")
    except Exception as e:
        st.error(f"❌ MACD 指标计算异常：{e}")

    # 计算 RSI
    try:
        rsi_series = ta.rsi(df['close'])
        if rsi_series is not None:
            df['RSI'] = rsi_series
        else:
            st.warning("⚠️ 无法计算 RSI 指标")
    except Exception as e:
        st.error(f"❌ RSI 指标计算异常：{e}")

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
