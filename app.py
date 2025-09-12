import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak
import plotly.graph_objects as go
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError

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
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        st.error(f"❌ AkShare 获取数据失败：{e}")
        return pd.DataFrame()

def analyze_tech(df):
    if 'close' not in df.columns or df['close'].isna().all():
        st.error("❌ 技术指标计算失败：未找到有效的收盘价数据")
        return df
    try:
        # 指标计算
        macd_df = ta.macd(df['close'])
        boll_df = ta.bbands(df['close'])
        df = pd.concat([df, macd_df, boll_df], axis=1)

        # 正确重命名
        df.rename(columns={
            'MACD_12_26_9': 'MACD',
            'MACDs_12_26_9': 'MACD_signal',
            'MACDh_12_26_9': 'MACD_hist',
            'BBL_20_2.0': 'BOLL_L',
            'BBM_20_2.0': 'BOLL_M',
            'BBU_20_2.0': 'BOLL_U',
        }, inplace=True)

        # RSI
        df['RSI'] = ta.rsi(df['close'])

        # 均线
        df['MA5'] = ta.sma(df['close'], length=5)
        df['MA10'] = ta.sma(df['close'], length=10)
        df['MA20'] = ta.sma(df['close'], length=20)

        # 金叉死叉标注
        df['buy_signal'] = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
        df['sell_signal'] = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))

    except Exception as e:
        st.error(f"❌ 技术指标计算异常：{e}")
        return df

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

        st.subheader("📉 图表分析展示")
        tab1, tab2, tab3 = st.tabs(["K线+均线+BOLL", "MACD", "RSI"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], mode='lines', name='MA5'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA10'], mode='lines', name='MA10'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], mode='lines', name='MA20'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['BOLL_U'], mode='lines', name='BOLL上轨'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['BOLL_M'], mode='lines', name='BOLL中轨'))
            fig.add_trace(go.Scatter(x=df['date'], y=df['BOLL_L'], mode='lines', name='BOLL下轨'))
            fig.add_trace(go.Scatter(x=df[df['buy_signal']]['date'], y=df[df['buy_signal']]['close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='买入点'))
            fig.add_trace(go.Scatter(x=df[df['sell_signal']]['date'], y=df[df['sell_signal']]['close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='卖出点'))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
            macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], name='Signal', line=dict(color='orange')))
            macd_fig.add_trace(go.Bar(x=df['date'], y=df['MACD_hist'], name='Histogram'))
            st.plotly_chart(macd_fig, use_container_width=True)

        with tab3:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI'))
            rsi_fig.add_hline(y=70, line_dash='dash', line_color='red')
            rsi_fig.add_hline(y=30, line_dash='dash', line_color='green')
            st.plotly_chart(rsi_fig, use_container_width=True)

        st.subheader("🧠 ChatGPT 策略建议")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)
else:
    st.info("请输入6位股票代码，例如 000001 或 600519")
