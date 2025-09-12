
import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak
import plotly.graph_objects as go
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError

# 配置 API 和页面
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="AkShare + ChatGPT 股票分析", layout="wide")
st.title("\ud83d\udcc8 AkShare + ChatGPT 技术面股票分析")

# 获取数据
def fetch_ak_kline(code):
    if len(code) != 6:
        st.error("\u80a1\u7968\u4ee3\u7801\u5e94\u4e3a6\u4f4d\u6570\u5b57\uff0c\u4f8b\u5982 000001 或 600519")
        return pd.DataFrame()
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20220101", adjust="qfq")
        df.rename(columns={"\u65e5\u671f": "date", "\u6536\u76d8": "close", "\u5f00\u76d8": "open", "\u6700\u9ad8": "high", "\u6700\u4f4e": "low", "\u6210\u4ea4\u91cf": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        st.error(f"\u274c AkShare \u83b7\u53d6\u6570\u636e\u5931\u8d25: {e}")
        return pd.DataFrame()

# 技术指标
def analyze_tech(df):
    df.ta.macd(close='close', append=True)
    df.ta.rsi(close='close', append=True)
    df.ta.bbands(close='close', append=True)
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=60, append=True)
    return df

# 第一个图 - K线+MA+BOLL

def plot_candlestick(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_5'], mode='lines', name='MA5'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_20'], mode='lines', name='MA20'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_60'], mode='lines', name='MA60'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['BBL_5_2.0'], mode='lines', name='BOLL Low'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['BBM_5_2.0'], mode='lines', name='BOLL Mid'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['BBU_5_2.0'], mode='lines', name='BOLL Up'))
    fig.update_layout(height=600, title='K线图+MA+BOLL')
    return fig

# MACD

def plot_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['date'], y=df['MACDh_12_26_9'], name='Histogram'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_12_26_9'], name='MACD'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACDs_12_26_9'], name='Signal'))
    fig.update_layout(height=400, title='MACD 指标')
    return fig

# RSI

def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI_14'], name='RSI'))
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.update_layout(height=400, title='RSI 指标')
    return fig

# ChatGPT 分析

def explain_by_gpt(stock_code, row):
    prompt = f"""
    你是技术面股票分析师，请根据以下技术指标数据给出简明策略建议:

股票代码：{stock_code}
技术指标：
{row.to_string()}

输出示例：
【操作建议】买入/持有/观望/出售  （原因）
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
        return "\u274c ChatGPT \u8bf7\u6c42\u8fc7\u4e8e\u9891\u7e41\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5"
    except AuthenticationError:
        return "\u274c OpenAI API \u5bc6\u94a5\u9519\u8bef\u6216\u5931\u6548"
    except OpenAIError as e:
        return f"\u274c OpenAI \u9519误: {str(e)}"
    except Exception as e:
        return f"\u274c \u7cfb\u7edf\u9519\u8bef: {str(e)}"

# 主逻辑
stock_code = st.text_input("\u8bf7输\u5165 6 \u4f4d\u80a1\u7968\u4ee3\u7801（\u4e0d\u5e26 SH/SZ 前缀\uff09")
if stock_code:
    with st.spinner("\u83b7\u53d6数\u636e\u5206析中..."):
        df = fetch_ak_kline(stock_code)
        if df.empty:
            st.stop()
        df = analyze_tech(df)
        last_row = df.iloc[-1]

        st.subheader("\ud83d\udcca \u6700\u8fd1\u884c\u60c5与\u6280\u672f\u6307\u6807")
        st.dataframe(df.tail(5)[["date", "close", "MACD_12_26_9", "MACDs_12_26_9", "RSI_14"]].set_index("date"))

        # 分页图表
        tab1, tab2, tab3 = st.tabs(["\ud83d\udd39 K线 + MA + BOLL", "\ud83c\udfaf MACD", "\ud83c\udf10 RSI"])
        with tab1:
            st.plotly_chart(plot_candlestick(df), use_container_width=True)
        with tab2:
            st.plotly_chart(plot_macd(df), use_container_width=True)
        with tab3:
            st.plotly_chart(plot_rsi(df), use_container_width=True)

        st.subheader("\ud83e\udde0 ChatGPT \u7b56\u7565\u5efa\u8bae")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)
else:
    st.info("\u8bf7输\u5165 6 \u4f4d\u80a1\u7968\u4ee3\u7801")
