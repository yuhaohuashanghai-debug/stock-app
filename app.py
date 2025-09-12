
import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak
import plotly.graph_objects as go
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError

# ✅ 设置 OpenAI 密钥
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ 页面初始化
st.set_page_config(page_title="AkShare + ChatGPT 股票分析", layout="wide")
st.title("📈 AkShare + ChatGPT 技术面股票分析")

# ✅ 获取行情数据
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

# ✅ 技术指标分析
def analyze_tech(df):
    if 'close' not in df.columns or df['close'].isna().all():
        st.error("❌ 技术指标计算失败：未找到有效的收盘价数据")
        return df
    try:
        macd_df = ta.macd(df['close'])
        boll_df = ta.bbands(df['close'])
        df = pd.concat([df, macd_df], axis=1)

        if boll_df is not None and all(col in boll_df.columns for col in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
            df = pd.concat([df, boll_df], axis=1)
            df.rename(columns={
                'BBL_20_2.0': 'BOLL_L',
                'BBM_20_2.0': 'BOLL_M',
                'BBU_20_2.0': 'BOLL_U',
            }, inplace=True)
        else:
            st.warning("⚠️ 布林带指标计算失败，部分图表可能无法显示")

        df.rename(columns={
            'MACD_12_26_9': 'MACD',
            'MACDs_12_26_9': 'MACD_signal',
            'MACDh_12_26_9': 'MACD_hist',
        }, inplace=True)

        df['RSI'] = ta.rsi(df['close'])
        df['MA5'] = ta.sma(df['close'], length=5)
        df['MA10'] = ta.sma(df['close'], length=10)
        df['MA20'] = ta.sma(df['close'], length=20)

        df['buy_signal'] = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
        df['sell_signal'] = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))

    except Exception as e:
        st.error(f"❌ 技术指标计算异常：{e}")
    return df

# ✅ ChatGPT 生成策略建议
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

# ✅ 用户输入股票代码
stock_code = st.text_input("请输入股票代码（6位，不带 SH/SZ 后缀）如 600519:")

if stock_code:
    with st.spinner("正在获取数据和分析中..."):
        df = fetch_ak_kline(stock_code)
        if df.empty:
            st.stop()

        df = analyze_tech(df)
        last_row = df.iloc[-1]

        # ✅ 显示最近数据表
        st.subheader("📊 最近行情与技术指标")
        st.dataframe(df.tail(5)[['date', 'close', 'MACD', 'MACD_signal', 'RSI']].set_index('date'))

        # ✅ 图表可视化分页
        st.subheader("📉 图表分析展示")
        chart_tab = st.tabs(["K线+均线+BOLL", "MACD", "RSI 相对强弱指标"])

        with chart_tab[0]:
            try:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['date'], open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='K线'))

                for ma, color in zip(['MA5', 'MA10', 'MA20'], ['blue', 'orange', 'green']):
                    if ma in df.columns:
                        fig.add_trace(go.Scatter(x=df['date'], y=df[ma], mode='lines', name=ma, line=dict(color=color)))

                for boll, color in zip(['BOLL_U', 'BOLL_M', 'BOLL_L'], ['red', 'gray', 'red']):
                    if boll in df.columns:
                        fig.add_trace(go.Scatter(x=df['date'], y=df[boll], mode='lines', name=boll, line=dict(color=color, dash='dot')))

                if 'buy_signal' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[df['buy_signal']]['date'], y=df[df['buy_signal']]['close'],
                        mode='markers', name='买入信号', marker=dict(color='green', size=10, symbol='circle')))

                if 'sell_signal' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[df['sell_signal']]['date'], y=df[df['sell_signal']]['close'],
                        mode='markers', name='卖出信号', marker=dict(color='red', size=10, symbol='circle')))

                fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ K线图绘制失败：{e}")

        with chart_tab[1]:
            try:
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
                macd_fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], name='Signal', line=dict(color='orange')))
                macd_fig.add_trace(go.Bar(x=df['date'], y=df['MACD_hist'], name='Histogram'))
                macd_fig.update_layout(height=400, margin=dict(t=10, b=10))
                st.plotly_chart(macd_fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ MACD 图绘制失败：{e}")

        with chart_tab[2]:
            try:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='purple')))
                rsi_fig.add_shape(type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1], y0=70, y1=70, line=dict(color="red", dash="dash"))
                rsi_fig.add_shape(type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1], y0=30, y1=30, line=dict(color="green", dash="dash"))
                rsi_fig.update_layout(height=400, margin=dict(t=10, b=10))
                st.plotly_chart(rsi_fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ RSI 图绘制失败：{e}")

        # ✅ GPT 策略建议展示
        st.subheader("🧠 ChatGPT 策略建议")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)
else:
    st.info("请输入6位股票代码，例如 000001 或 600519")
