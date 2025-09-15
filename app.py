import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 智能趋势点评")

# ===== 字段万能适配（核心模块） =====
FIELD_MAP = {
    'code': ['代码', '股票代码', '证券代码', 'con_code', '基金代码', 'symbol'],
    'name': ['名称', '股票名称', '证券名称', '成分券名称', '基金简称', 'display_name', 'name'],
    'change_pct': ['涨跌幅', '涨幅', '涨跌幅度', 'changepercent', '涨幅(%)'],
    'board': ['板块名称', '概念名称', '行业名称'],
}

def get_col(df, key, strict=True, verbose=False):
    candidates = FIELD_MAP.get(key, [key])
    # 优先严格全匹配
    for cand in candidates:
        for col in df.columns:
            if cand.strip().lower() == col.strip().lower():
                if verbose:
                    print(f"找到{key}字段: {col}")
                return col
    # 再用“包含”模糊匹配
    for cand in candidates:
        for col in df.columns:
            if cand.strip().lower() in col.strip().lower() or col.strip().lower() in cand.strip().lower():
                if verbose:
                    print(f"模糊匹配到{key}字段: {col}")
                return col
    if not strict:
        return None
    raise KeyError(f"DataFrame找不到‘{key}’相关字段，实际字段为: {df.columns.tolist()}")

@st.cache_data(show_spinner=False)
def get_all_a_codes():
    stock_df = ak.stock_info_a_code_name()
    code_col = get_col(stock_df, 'code')
    return stock_df[code_col].tolist()

@st.cache_data(show_spinner=False)
def get_all_etf_codes():
    etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    code_col = get_col(etf_df, 'code')
    return etf_df[code_col].tolist()

@st.cache_data(show_spinner=False)
def get_index_codes(index_code):
    # symbol参数，不是index
    df = ak.index_stock_cons(symbol=index_code)
    code_col = get_col(df, 'code')
    return df[code_col].tolist()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_industry_boards(topn=20):
    try:
        df = ak.stock_board_industry_name_ths()
        change_col = get_col(df, 'change_pct', strict=False)
        board_col = get_col(df, 'board', strict=False)
        if change_col:
            hot_df = df.sort_values(change_col, ascending=False).head(topn)
        else:
            hot_df = df.head(topn)
        if board_col and change_col:
            return hot_df[[board_col, change_col]]
        elif board_col:
            return hot_df[[board_col]]
        else:
            return hot_df
    except Exception as e:
        print(f"获取行业板块异常: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_name_ths()
        change_col = get_col(df, 'change_pct', strict=False)
        board_col = get_col(df, 'board', strict=False)
        if change_col:
            hot_df = df.sort_values(change_col, ascending=False).head(topn)
        else:
            hot_df = df.head(topn)
        if board_col and change_col:
            return hot_df[[board_col, change_col]]
        elif board_col:
            return hot_df[[board_col]]
        else:
            return hot_df
    except Exception as e:
        print(f"获取概念板块异常: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_etf_boards(topn=20):
    try:
        df = ak.fund_etf_spot_em()
        change_col = get_col(df, 'change_pct', strict=False)
        name_col = get_col(df, 'name', strict=False)
        code_col = get_col(df, 'code', strict=False)
        if change_col:
            hot_df = df.sort_values(change_col, ascending=False).head(topn)
        else:
            hot_df = df.head(topn)
        cols = [c for c in [code_col, name_col, change_col] if c]
        return hot_df[cols] if cols else hot_df
    except Exception as e:
        print(f"获取ETF排行异常: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_board_stocks(board_name):
    try:
        df = ak.stock_board_concept_cons_ths(symbol=board_name)
        code_col = get_col(df, 'code')
        return df[code_col].tolist()
    except Exception as e:
        print(f"获取板块成分股异常: {e}")
        return []

# ========== 结果导出为Excel（修正download_button的to_excel问题） ==========
def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.read()

# ========== 其余主流程与原来一致，接口替换上述函数即可 ==========

# ...（后续主逻辑略，直接调用get_all_a_codes等新函数）

# ====== 举例如何修正DataFrame导出（选股结果导出Excel） ======
# st.download_button(
#     "导出全部明细为Excel",
#     data=to_excel_bytes(df_sel),  # 用此函数，不再直接df.to_excel()
#     file_name="AI选股明细.xlsx"
# )

