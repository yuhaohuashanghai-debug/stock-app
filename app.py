import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 智能趋势点评")

# ============ 通用工具函数 =============
def get_first_valid_column(df, candidates):
    """在候选字段中返回第一个存在的列"""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"字段未找到，现有字段: {df.columns.tolist()}, 候选: {candidates}")

def get_code_list(df):
    code_candidates = ["code", "symbol", "基金代码", "代码", "con_code", "成分券代码"]
    code_col = get_first_valid_column(df, code_candidates)
    return df[code_col].tolist()

def get_name_list(df):
    name_candidates = ["name", "名称", "证券简称", "display_name"]
    name_col = get_first_valid_column(df, name_candidates)
    return df[name_col].tolist()

# ============ 选股池工具函数 =============
@st.cache_data(ttl=1800)
def get_all_a_codes():
    stock_df = ak.stock_info_a_code_name()
    return get_code_list(stock_df)

@st.cache_data(ttl=1800)
def get_all_etf_codes():
    etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    return get_code_list(etf_df)

@st.cache_data(ttl=1800)
def get_index_codes(index_code):
    df = ak.index_stock_cons(symbol=index_code)
    return get_code_list(df)

@st.cache_data(ttl=300)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_name_ths()
        return df.head(topn)
    except Exception as e:
        st.error(f"获取概念板块失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_hot_industry_boards(topn=20):
    try:
        df = ak.stock_board_industry_name_ths()
        return df.head(topn)
    except Exception as e:
        st.error(f"获取行业板块失败: {e}")
        return pd.DataFrame()

# ============ 主体部分 =============
st.subheader("选股池测试")

market_pool = st.selectbox("选择市场池", ["全A股", "全ETF", "沪深300"])

if market_pool == "全A股":
    codes = get_all_a_codes()
elif market_pool == "全ETF":
    codes = get_all_etf_codes()
elif market_pool == "沪深300":
    codes = get_index_codes("000300")

st.write(f"加载股票数量: {len(codes)}")

# 演示表格，注意替换 use_container_width
if len(codes) > 0:
    demo_df = pd.DataFrame({"股票代码": codes[:10]})
    st.dataframe(demo_df, width="stretch")  # ✅ 修复 use_container_width
