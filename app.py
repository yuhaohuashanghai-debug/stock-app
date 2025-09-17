import streamlit as st
import pandas as pd
import akshare as ak
import pandas_ta as ta
import plotly.graph_objects as go

# ========== 页面配置 ==========
st.set_page_config(page_title="📈 实时股票AI分析平台", layout="wide")
st.title("📊 实时股票技术分析 + 板块概念联动")

# ========== 工具函数 ==========
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kline(code: str, start_date="20220101"):
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start_date, adjust="qfq"
        )
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def em_concepts_topn(topn=200):
    df = ak.stock_board_concept_name_em()
    return df[["板块名称", "板块代码"]].head(topn)

@st.cache_data(ttl=3600, show_spinner=False)
def em_concept_members(concept_name: str) -> pd.DataFrame:
    try:
        cdf = ak.stock_board_concept_cons_em(symbol=concept_name)
        rename_map = {"代码": "code", "股票代码": "code", "名称": "name", "股票名称": "name"}
        cdf.rename(columns={k: v for k, v in rename_map.items() if k in cdf.columns}, inplace=True)
        cdf["code"] = cdf["code"].astype(str).str[-6:]
        return cdf[["code", "name"]].drop_duplicates()
    except Exception:
        return pd.DataFrame(columns=["code", "name"])

@st.cache_data(ttl=3600, show_spinner=False)
def concepts_by_stock(code: str, topn=200):
    code = str(code).zfill(6)
    cons = em_concepts_topn(topn=topn)
    hits = []
    for cname in cons["板块名称"].tolist():
        mem = em_concept_members(cname)
        if not mem.empty and (mem["code"] == code).any():
            hits.append(cname)
    return hits

# ========== 页面布局 ==========
tab1, tab2, tab3 = st.tabs(["📈 K线图", "📊 技术指标", "📊 板块概念联动"])

with tab1:
    st.subheader("📈 股票K线展示")
    code = st.text_input("输入股票代码（6位数字，例如 000001）", "000001")
    df = fetch_kline(code)
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df["date"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="K线"
        )])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("未能获取该股票数据，请检查代码或接口。")

with tab2:
    st.subheader("📊 技术指标分析")
    if not df.empty:
        df["MA5"] = df["close"].rolling(5).mean()
        df["RSI"] = ta.rsi(df["close"], length=14)
        st.line_chart(df.set_index("date")[["close", "MA5"]])
        st.line_chart(df.set_index("date")[["RSI"]])
    else:
        st.info("请先在左边输入有效的股票代码。")

with tab3:
    st.subheader("📊 板块概念联动分析")
    if df.empty:
        st.info("请先输入股票代码")
    else:
        concepts = concepts_by_stock(code, topn=200)
        if concepts:
            st.write(f"股票 {code} 所属概念：", "、".join(concepts))
        else:
            st.warning("未找到所属概念，可能接口返回为空或网络波动。")
