import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError

# ✅ 设置 OpenAI 密钥
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ 页面初始化
st.set_page_config(page_title="AkShare + ChatGPT 股票分析", layout="wide")
st.title("📈 AkShare + ChatGPT 技术面股票分析")

# ✅ 获取行情数据
@st.cache_data(ttl=3600)
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
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception as e:
        st.error(f"❌ AkShare 获取数据失败：{e}")
        return pd.DataFrame()

# ✅ 技术指标分析
@st.cache_data(ttl=3600)
def analyze_tech(df):
    if 'close' not in df.columns or df['close'].isna().all():
        st.error("❌ 技术指标计算失败：未找到有效的收盘价数据")
        return df
    try:
        macd_df = ta.macd(df['close'])
        boll_df = ta.bbands(df['close'])
        df = pd.concat([df, macd_df, boll_df], axis=1)

        df.rename(columns={
            'MACD_12_26_9': 'MACD',
            'MACDs_12_26_9': 'MACD_signal',
            'MACDh_12_26_9': 'MACD_hist',
            'BBL_20_2.0': 'BOLL_L',
            'BBM_20_2.0': 'BOLL_M',
            'BBU_20_2.0': 'BOLL_U',
        }, inplace=True)

        df['RSI'] = ta.rsi(df['close'])
        df['MA5'] = ta.sma(df['close'], length=5)
        df['MA10'] = ta.sma(df['close'], length=10)
        df['MA20'] = ta.sma(df['close'], length=20)

        df['buy_signal'] = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
        df['sell_signal'] = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))

        df = df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ 技术指标计算异常：{e}")
    return df

# ✅ 策略回测模块
def backtest_signals(df, hold_days=5):
    results = []
    for i in df[df['buy_signal']].index:
        if i + hold_days < len(df):
            future_return = (df.loc[i+hold_days, 'close'] - df.loc[i, 'close']) / df.loc[i, 'close']
            results.append(future_return)
    if results:
        win_rate = sum(r > 0 for r in results) / len(results)
        return {"样本数": len(results), "平均涨幅": round(sum(results)/len(results)*100, 2), "胜率": round(win_rate*100, 2)}
    return {"样本数": 0, "平均涨幅": 0, "胜率": 0}

# ✅ ChatGPT 生成策略建议
def explain_by_gpt(stock_code, row):
    prompt = f"""
你是一名技术面分析师，请根据以下股票的技术指标给出简明逻辑策略建议（偏向短线操作）：

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

        # ✅ 图表数据滑块
        st.subheader("📉 图表分析展示")
        max_days = len(df)
        default_days = min(90, max_days)
        window = st.slider("请选择图表展示周期（天数）:", min_value=30, max_value=max_days, value=default_days, step=10)
        df_plot = df[df['date'] > df['date'].max() - pd.Timedelta(days=window)]

        chart_tab = st.tabs(["K线图", "MACD", "RSI", "成交量"])

        with chart_tab[0]:
            try:
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Candlestick(x=df_plot['date'], open=df_plot['open'], high=df_plot['high'],
                                             low=df_plot['low'], close=df_plot['close'], name='K线'))
                for ma, color in zip(['MA5', 'MA10', 'MA20'], ['blue', 'orange', 'green']):
                    if ma in df_plot.columns:
                        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[ma], mode='lines', name=ma,
                                                 line=dict(color=color)))
                for boll, color in zip(['BOLL_U', 'BOLL_M', 'BOLL_L'], ['red', 'gray', 'red']):
                    if boll in df_plot.columns:
                        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[boll], mode='lines', name=boll,
                                                 line=dict(color=color, dash='dot')))
                fig.add_trace(go.Scatter(x=df_plot[df_plot['buy_signal']]['date'],
                                         y=df_plot[df_plot['buy_signal']]['close'], mode='markers', name='买入信号',
                                         marker=dict(color='green', size=10, symbol='triangle-up')))
                fig.add_trace(go.Scatter(x=df_plot[df_plot['sell_signal']]['date'],
                                         y=df_plot[df_plot['sell_signal']]['close'], mode='markers', name='卖出信号',
                                         marker=dict(color='red', size=10, symbol='triangle-down')))
                fig.update_layout(height=600, hovermode='x unified', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ K线图绘制失败：{e}")

        with chart_tab[1]:
            try:
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['MACD'], name='MACD', line=dict(color='blue')))
                macd_fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['MACD_signal'], name='Signal', line=dict(color='orange')))
                macd_fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['MACD_hist'], name='Histogram'))
                macd_fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(macd_fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ MACD 图绘制失败：{e}")

        with chart_tab[2]:
            try:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['RSI'], name='RSI', line=dict(color='purple')))
                rsi_fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=70, y1=70,
                                  line=dict(color="red", dash="dash"))
                rsi_fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=30, y1=30,
                                  line=dict(color="green", dash="dash"))
                rsi_fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(rsi_fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ RSI 图绘制失败：{e}")

        with chart_tab[3]:
            try:
                vol_fig = go.Figure()
                vol_fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['volume'], name='成交量', marker_color='lightblue'))
                vol_fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['volume'].rolling(5).mean(), name='5日均量', line=dict(color='orange')))
                vol_fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(vol_fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ 成交量图绘制失败：{e}")

        st.subheader("📈 策略信号回测结果")
        backtest_result = backtest_signals(df, hold_days=5)
        st.write(f"买入信号样本数: {backtest_result['样本数']}")
        st.write(f"平均 5 日涨幅: {backtest_result['平均涨幅']}%")
        st.write(f"胜率: {backtest_result['胜率']}%")

        st.subheader("🧠 ChatGPT 策略建议")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)
else:
    st.info("请输入6位股票代码，例如 000001 或 600519")
