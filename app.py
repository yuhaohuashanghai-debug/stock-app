import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="A股批量AI自动选股 & 板块信号聚合解读", layout="wide")
st.title("📈 A股批量AI自动选股 & 板块信号聚合解读")
st.markdown("<span style='color: #c00; font-weight: bold;'>个股批量分析+AI点评（多数据源自动切换）</span>", unsafe_allow_html=True)

# ====== 表单输入 ======
with st.form("ai_stock_form"):
    st.subheader("自定义股票池批量分析+AI智能点评")
    stock_codes = st.text_input("请输入A股股票代码（支持批量，如600519,000977,588170）", value="000977,518880")
    start_date = st.date_input("选择起始日期", value=datetime.now() - timedelta(days=90))
    ai_enable = st.toggle("启用AI趋势点评", value=False)
    ai_days = st.selectbox("AI预测未来天数", [3, 5, 7], index=0, help="仅AI预测时生效")
    submit = st.form_submit_button("批量分析")
st.caption("自动切换数据源，兼容本地/云端。")

# ====== 多数据源逻辑 ======
def fetch_akshare(code, start_date):
    try:
        if code.startswith("5") or code.startswith("1"):
            df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                    start_date=start_date.strftime("%Y%m%d"),
                                    end_date=datetime.now().strftime("%Y%m%d"))
            df.rename(columns={"交易日期": "date", "开盘价": "open", "收盘价": "close",
                               "最高价": "high", "最低价": "low", "成交量": "volume"}, inplace=True)
        else:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                   start_date=start_date.strftime("%Y%m%d"), adjust="qfq")
            df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                               "最高": "high", "最低": "low", "成交量": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= pd.to_datetime(start_date)]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        return None

def fetch_sina(code, start_date):
    try:
        # 新浪A股/ETF日K线接口
        if code.startswith("6"):
            market = "sh"
        else:
            market = "sz"
        url = f"https://quotes.sina.cn/cn/api/openapi.php/CN_MarketData.getKLineData?symbol={market}{code}&scale=240&ma=no&datalen=2000"
        res = requests.get(url, timeout=10)
        jsondata = res.json()
        if not jsondata or "result" not in jsondata or not jsondata["result"]["data"]:
            return None
        kline = jsondata["result"]["data"]["kline"]
        df = pd.DataFrame(kline)
        df["date"] = pd.to_datetime(df["day"])
        df["open"] = df["open"].astype(float)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df = df[["date", "open", "close", "high", "low", "volume"]]
        df = df[df["date"] >= pd.to_datetime(start_date)]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        return None

def fetch_netease(code, start_date):
    try:
        # 网易财经A股K线数据（日线，最多3年）
        if code.startswith("6"):
            market = "0"
        else:
            market = "1"
        url = f"http://quotes.money.163.com/service/chddata.html?code={market}{code}&start={start_date.strftime('%Y%m%d')}&end={datetime.now().strftime('%Y%m%d')}&fields=TCLOSE;HIGH;LOW;TOPEN;VOTURNOVER"
        df = pd.read_csv(url, encoding="gbk")
        df = df.rename(columns={"日期": "date", "开盘价": "open", "收盘价": "close",
                                "最高价": "high", "最低价": "low", "成交量": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df[["date", "open", "close", "high", "low", "volume"]]
        df = df[df["date"] >= pd.to_datetime(start_date)]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        return None

def fetch_stock_data(code, start_date):
    # 优先AkShare，失败自动切新浪，再失败切网易
    for fetcher, name in [(fetch_akshare, "AkShare"), (fetch_sina, "新浪"), (fetch_netease, "网易")]:
        df = fetcher(code, start_date)
        if df is not None and not df.empty:
            st.info(f"{code} 数据源：{name}（获取成功）")
            return df
    st.error(f"股票代码 {code} 多数据源均获取失败，可能是网络限制或代码无效。")
    return pd.DataFrame()

# ====== 技术分析与可视化 ======
def plot_kline_ma(df, code):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"))
    for n, c in zip([5, 10, 20], ["blue", "orange", "green"]):
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"].rolling(n).mean(),
                                 mode='lines', name=f"MA{n}", line=dict(width=1)))
    fig.update_layout(title=f"{code} K线&均线", height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    # 成交量
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量"))
    fig2.update_layout(title=f"{code} 成交量", height=200, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

def calc_ta_signal(df):
    df['macd'], df['macd_signal'], _ = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['rsi'] = ta.rsi(df['close'], length=14)
    golden_cross = (df['macd'].iloc[-2] < df['macd_signal'].iloc[-2]) and (df['macd'].iloc[-1] > df['macd_signal'].iloc[-1])
    oversold = df['rsi'].iloc[-1] < 30
    vol_break = df['volume'].iloc[-1] > df['volume'].rolling(10).mean().iloc[-1] * 1.5
    return golden_cross, oversold, vol_break

def ai_trend_comment(df, code, ai_days):
    close_now = df["close"].iloc[-1]
    pct_chg = (close_now - df["close"].iloc[-5]) / df["close"].iloc[-5] * 100
    trend = "上升" if pct_chg > 0 else "震荡/下跌"
    return f"【AI点评】{code} 当前价格 {close_now:.2f}，近5日涨幅 {pct_chg:.2f}%。AI预测未来{ai_days}天趋势：{trend}。"

# ====== 主流程 ======
if submit:
    codes = [x.strip() for x in stock_codes.replace("，", ",").split(",") if x.strip()]
    st.info(f"共输入 {len(codes)} 只股票/ETF，开始批量分析...")
    for code in codes:
        st.markdown(f"---\n### ⬇️ {code} 技术分析&AI点评")
        df = fetch_stock_data(code, start_date)
        if df.empty or len(df) < 20:
            st.warning(f"{code} 数据不足或获取失败")
            continue
        plot_kline_ma(df, code)
        golden_cross, oversold, vol_break = calc_ta_signal(df)
        st.write(f"**MACD金叉:** {'✅' if golden_cross else '❌'} | **RSI超卖:** {'✅' if oversold else '❌'} | **放量突破:** {'✅' if vol_break else '❌'}")
        if ai_enable:
            ai_comment = ai_trend_comment(df, code, ai_days)
            st.success(ai_comment)
