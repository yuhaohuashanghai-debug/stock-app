import streamlit as st
import pandas as pd
import akshare as ak
import pandas_ta as ta
import plotly.graph_objects as go
import io

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 智能趋势点评")

# ============ 容错的 AKShare 数据拉取函数 ============

@st.cache_data(show_spinner=False)
def get_all_a_codes():
    df = ak.stock_info_a_code_name()
    # 兼容不同字段
    code_col = "证券代码" if "证券代码" in df.columns else "code"
    return df[code_col].tolist()

@st.cache_data(show_spinner=False)
def get_all_etf_codes():
    df = ak.fund_etf_category_sina(symbol="ETF基金")
    # 兼容不同字段
    if "代码" in df.columns:
        return df["代码"].tolist()
    elif "symbol" in df.columns:
        return df["symbol"].tolist()
    else:
        st.error("ETF数据接口字段异常")
        return []

@st.cache_data(show_spinner=False)
def get_index_codes(symbol):
    try:
        df = ak.index_stock_cons(symbol=symbol)
        # 兼容不同字段
        if "con_code" in df.columns:
            return df["con_code"].tolist()
        elif "成分券代码" in df.columns:
            return df["成分券代码"].tolist()
        else:
            st.warning("指数成分股数据字段异常！")
            return []
    except Exception as e:
        st.error(f"指数成分股数据获取失败: {e}")
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_name_ths()
        # 排序容错，涨跌幅有多个名字
        pct_col = "涨跌幅" if "涨跌幅" in df.columns else ("涨跌幅(%)" if "涨跌幅(%)" in df.columns else None)
        if pct_col:
            df = df.sort_values(pct_col, ascending=False)
        return df.head(topn)
    except Exception as e:
        st.error(f"概念板块排行获取失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_industry_boards(topn=20):
    try:
        df = ak.stock_board_industry_name_ths()
        # 排序容错
        pct_col = "涨跌幅" if "涨跌幅" in df.columns else ("涨跌幅(%)" if "涨跌幅(%)" in df.columns else None)
        if pct_col:
            df = df.sort_values(pct_col, ascending=False)
        return df.head(topn)
    except Exception as e:
        st.error(f"行业板块排行获取失败: {e}")
        return pd.DataFrame()

# ============ 技术指标批量计算示例 ============

@st.cache_data(ttl=1800)
def fetch_kline_df(code, start_date="20230101"):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
        # 字段适配
        df.rename(columns={
            "日期":"date", "开盘":"open", "收盘":"close", "最高":"high", "最低":"low", "成交量":"volume"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.warning(f"拉取K线失败: {e}")
        return pd.DataFrame()

def calc_signals(df):
    if df.empty or "close" not in df:
        return {}
    # 技术指标
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["macd"], df["macd_signal"], df["macd_hist"] = ta.macd(df["close"])
    df["rsi"] = ta.rsi(df["close"])
    # 信号示例
    last = df.iloc[-1]
    signal = []
    if last["macd"] > last["macd_signal"]:
        signal.append("MACD金叉")
    if last["rsi"] < 30:
        signal.append("RSI超卖")
    if last["close"] > last["ma5"] > last["ma20"]:
        signal.append("均线上穿")
    return {
        "信号": " / ".join(signal) if signal else "无显著信号",
        "MA5": round(last["ma5"],2) if pd.notna(last["ma5"]) else None,
        "MA20": round(last["ma20"],2) if pd.notna(last["ma20"]) else None,
        "MACD": round(last["macd"],2) if pd.notna(last["macd"]) else None,
        "RSI": round(last["rsi"],2) if pd.notna(last["rsi"]) else None,
    }

# ============ 页面逻辑 ============

# 选股池选择
market_pool = st.sidebar.selectbox("选股池", ["全A股", "全ETF", "沪深300", "科创50", "热门行业板块", "热门概念板块"])
if market_pool == "全A股":
    codes = get_all_a_codes()
elif market_pool == "全ETF":
    codes = get_all_etf_codes()
elif market_pool == "沪深300":
    codes = get_index_codes("000300")
elif market_pool == "科创50":
    codes = get_index_codes("000688")
elif market_pool == "热门行业板块":
    st.subheader("🔥 今日热门行业板块排行（涨幅前20）")
    hot_df = get_hot_industry_boards()
    st.dataframe(hot_df)
    st.stop()
elif market_pool == "热门概念板块":
    st.subheader("🔥 今日热门概念板块排行（涨幅前20）")
    hot_df = get_hot_concept_boards()
    st.dataframe(hot_df)
    st.stop()
else:
    codes = []

# 显示数量限制（避免接口超载）
code_num = st.slider("分析股票数量", 3, min(50, len(codes)), 10)
codes = codes[:code_num]

# 批量分析
st.markdown("### 批量技术面信号&AI点评")
result_table = []
for code in st.progress(list(range(len(codes))), description="分析中..."):
    df = fetch_kline_df(code)
    sig = calc_signals(df)
    name = code  # 可根据需要拉取名称
    result_table.append({"代码": code, "信号": sig.get("信号", ""), "MA5": sig.get("MA5"), "MA20": sig.get("MA20"), "MACD": sig.get("MACD"), "RSI": sig.get("RSI")})

df_result = pd.DataFrame(result_table)
st.dataframe(df_result, use_container_width=True)

# ============ 导出 Excel 按钮（必须用BytesIO） ============
output = io.BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    df_result.to_excel(writer, index=False)
st.download_button(
    label="导出明细为Excel",
    data=output.getvalue(),
    file_name="AI选股明细.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

