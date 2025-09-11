
import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak

openai.api_key = st.secrets["OPENAI_API_KEY"]

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

from openai import OpenAI, OpenAIError, RateLimitError, AuthenticationError

openai.api_key = st.secrets["OPENAI_API_KEY"]

def explain_by_gpt(stock_code, row):
    prompt = f"""
你是一名技术面分析师，请根据以下股票的技术指标给出简明逻辑策略建议：

股票代码：{stock_code}
分析数据如下：
{row.to_string()}

输出示例：
买入/持有/观望/卖出，理由（简短）
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 或 gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        return "❌ ChatGPT 请求过于频繁，请稍后重试。"

    except AuthenticationError:
        return "❌ OpenAI API 密钥错误或已失效，请检查 `secrets.toml` 中的设置。"

    except OpenAIError as e:
        return f"❌ OpenAI 请求失败：{str(e)}"

    except Exception as e:
        return f"❌ 系统错误：{str(e)}"


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
