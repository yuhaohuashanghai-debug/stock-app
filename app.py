
import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak
import plotly.graph_objects as go
from openai import OpenAI
from openai import RateLimitError, AuthenticationError, OpenAIError

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="AkShare + ChatGPT 股票分析", layout="wide")
st.title("📈 AkShare + ChatGPT 技术面股票分析")

def fetch_ak_kline(code):
    if len(code) != 6:
        st.error("股票代码应为6位数字，例如 000001 或 600519")
        return pd.DataFrame()
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20220101", adjust="qfq")
        df.rename(columns={"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low", "成交量": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df[["close", "open", "high", "low", "volume"]] = df[["close", "open", "high", "low", "volume"]].astype(float)
        return df
    except Exception as e:
        st.error(f"❌ AkShare 获取数据失败：{e}")
        return pd.DataFrame()

def analyze_tech(df):
    try:
        macd_df = ta.macd(df['close'])
        if macd_df is not None:
            df = pd.concat([df, macd_df], axis=1)
            df.rename(columns={
                'MACD_12_26_9': 'MACD',
                'MACDs_12_26_9': 'MACD_signal',
                'MACDh_12_26_9': 'MACD_hist'
            }, inplace=True)
        df['RSI'] = ta.rsi(df['close'])
        df[['MA5', 'MA20', 'MA60']] = df['close'].rolling(5), df['close'].rolling(20), df['close'].rolling(60)
        bbands = ta.bbands(df['close'])
        df = pd.concat([df, bbands], axis=1)
    except Exception as e:
        st.error(f"❌ 技术指标计算失败：{e}")
    return df

def explain_by_gpt(stock_code, row):
    prompt = f"""
你是一名技术面分析师，请根据以下股票的技术指标给出简明逻辑策略建议：

股票代码：{stock_code}
分析数据如下：
{row.to_string()}

输出示例：
买入/持有/观望/卖出，理由（简要）
"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
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

        # 📊 多图分页可视化模块
        tab1, tab2, tab3 = st.tabs(["K线 + 均线 + BOLL", "MACD", "RSI"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="K线"))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], name='MA60'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBL_5_2.0'], name='BOLL下轨', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBM_5_2.0'], name='BOLL中轨', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['BBU_5_2.0'], name='BOLL上轨', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
            macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], name='Signal', line=dict(color='orange')))
            macd_fig.add_trace(go.Bar(x=df['date'], y=df['MACD_hist'], name='Hist'))
            st.plotly_chart(macd_fig, use_container_width=True)

        with tab3:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI 指标'))
            rsi_fig.add_hline(y=70, line=dict(dash='dash', color='red'))
            rsi_fig.add_hline(y=30, line=dict(dash='dash', color='green'))
            st.plotly_chart(rsi_fig, use_container_width=True)

        st.subheader("🧠 ChatGPT 策略建议")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)
else:
    st.info("请输入6位股票代码，例如 000001 或 600519")
