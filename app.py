
import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak
import plotly.graph_objects as go

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
from openai import RateLimitError, AuthenticationError, OpenAIError

def explain_by_gpt(stock_code, last_row):
    prompt = f"""
你是一名技术面分析师，请根据以下股票的技术指标给出简明逻辑策略建议：

股票代码：{stock_code}
分析数据如下：
{last_row.to_string()}

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

stock_code = st.text_input("请输入股票代码（此处为6位，不带 SH/SZ 后缀）如 600519:")

if stock_code:
    with st.spinner("正在获取数据和分析中..."):
        df = fetch_ak_kline(stock_code)
        if df.empty:
            st.stop()

        df = analyze_tech(df)
        last_row = df.iloc[-1]

        st.subheader("📈 最近行情与技术指标")
        st.dataframe(df.tail(5)[['date', 'close', 'MACD', 'MACD_signal', 'RSI']].set_index('date'))

        # 图表模块
        st.subheader("🔺 K线图 + 成交量")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="K线"
        ))
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name="成交量", yaxis='y2', marker=dict(color='lightblue')))
        fig.update_layout(yaxis2=dict(title="成交量", overlaying='y', side='right'), height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔹 MACD 指标")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name="MACD", line=dict(color="blue")))
        macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], name="Signal", line=dict(color="orange")))
        macd_fig.add_trace(go.Bar(x=df['date'], y=df['MACD_hist'], name="Histogram"))
        macd_fig.update_layout(height=400)
        st.plotly_chart(macd_fig, use_container_width=True)

        st.subheader("🔹 RSI 指标")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name="RSI", line=dict(color="purple")))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(height=400)
        st.plotly_chart(rsi_fig, use_container_width=True)

        st.subheader("🧠 ChatGPT 策略建议")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)
else:
    st.info("请输入6位股票代码，例如 000001 或 600519")
