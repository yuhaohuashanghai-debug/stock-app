import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 智能趋势点评")

# ====== 通用字段兼容工具 ======
def get_first_valid_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"字段未找到，现有字段: {df.columns.tolist()}, 候选: {candidates}")

def get_code_list(df):
    code_candidates = ["symbol", "基金代码", "代码", "con_code", "成分券代码"]
    code_col = get_first_valid_column(df, code_candidates)
    return df[code_col].tolist()

def get_name_list(df):
    name_candidates = ["name", "基金简称", "简称", "股票名称", "成分券名称", "板块名称"]
    name_col = get_first_valid_column(df, name_candidates)
    return df[name_col].tolist()

def get_pct_chg_col(df):
    chg_candidates = ["涨跌幅", "涨幅", "变动率", "日涨幅"]
    return get_first_valid_column(df, chg_candidates)

def show_columns(df, name="DataFrame"):
    st.write(f"【{name} 字段】: {df.columns.tolist()}")

def sort_by_pct_chg(df, topn=20):
    try:
        col = get_pct_chg_col(df)
        return df.sort_values(col, ascending=False).head(topn)
    except Exception as e:
        st.warning(f"排序字段未找到：{e}")
        return df.head(topn)

def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()

# ====== AkShare自动兼容接口 ======
@st.cache_data(ttl=1800)
def get_all_a_codes():
    stock_df = ak.stock_info_a_code_name()
    return get_code_list(stock_df)

@st.cache_data(ttl=1800)
def get_all_etf_codes():
    etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    return get_code_list(etf_df)

@st.cache_data(ttl=1800)
def get_index_codes_auto(index_code):
    df = ak.index_stock_cons(symbol=index_code)
    code_candidates = ["con_code", "成分券代码"]
    code_col = get_first_valid_column(df, code_candidates)
    return df[code_col].tolist()

# ====== 板块排行（行业 + 概念自动兼容） ======
@st.cache_data(ttl=1800)
def get_hot_industry_boards(topn=20):
    try:
        df = ak.stock_board_industry_index_ths()
    except Exception:
        df = ak.stock_board_industry_name_ths()
    show_columns(df, "行业板块")
    return sort_by_pct_chg(df, topn=topn)

@st.cache_data(ttl=1800)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_index_ths()
    except Exception:
        df = ak.stock_board_concept_name_ths()
    show_columns(df, "概念板块")
    return sort_by_pct_chg(df, topn=topn)

# ====== 板块成分股 ======
@st.cache_data(ttl=300)
def get_board_stocks(board_name):
    try:
        df = ak.stock_board_concept_cons_ths(symbol=board_name)
    except Exception:
        try:
            df = ak.stock_board_industry_cons_ths(symbol=board_name)
        except Exception:
            return []
    return get_code_list(df) if not df.empty else []

# ====== K线与信号判别 ======
def fetch_ak_data(code, start_date):
    df = pd.DataFrame()
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date.strftime("%Y%m%d"), adjust="qfq")
        if not df.empty:
            df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                               "最高": "high", "最低": "low", "成交量": "volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    except Exception:
        pass
    try:
        df = ak.fund_etf_hist_sina(symbol=code)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= pd.to_datetime(start_date)]
            df = df.sort_values("date").reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()

def calc_indicators(df):
    if "close" not in df.columns or len(df) < 20:
        return df
    try:
        df["SMA_5"] = ta.sma(df["close"], length=5)
        df["SMA_10"] = ta.sma(df["close"], length=10)
        df["SMA_20"] = ta.sma(df["close"], length=20)
        macd = ta.macd(df["close"])
        if macd is not None and not macd.empty:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACDs"] = macd["MACDs_12_26_9"]
        df["RSI_6"] = ta.rsi(df["close"], length=6)
    except Exception:
        pass
    return df

def signal_with_explain(df):
    explain = []
    signals = []
    latest = df.iloc[-1]
    pre = df.iloc[-2] if len(df) >= 2 else latest
    if "SMA_5" in df.columns and "SMA_10" in df.columns:
        if pre["SMA_5"] < pre["SMA_10"] and latest["SMA_5"] > latest["SMA_10"]:
            signals.append("5日均线金叉10日均线")
            explain.append("【均线金叉】：今日5日均线上穿10日均线（金叉），多头信号。")
        else:
            explain.append(f"【均线金叉】：5日均线({latest['SMA_5']:.2f}) {'>' if latest['SMA_5']>latest['SMA_10'] else '<='} 10日均线({latest['SMA_10']:.2f})，未发生金叉。")
    else:
        explain.append("【均线金叉】：数据不足，无法判断。")
    if "MACD" in df.columns and "MACDs" in df.columns:
        if pre["MACD"] < pre["MACDs"] and latest["MACD"] > latest["MACDs"]:
            signals.append("MACD金叉")
            explain.append("【MACD金叉】：今日MACD线上穿信号线，金叉出现，多头信号。")
        else:
            explain.append(f"【MACD金叉】：MACD({latest['MACD']:.3f}) {'>' if latest['MACD']>latest['MACDs'] else '<='} 信号线({latest['MACDs']:.3f})，未发生金叉。")
    else:
        explain.append("【MACD金叉】：数据不足，无法判断。")
    if "RSI_6" in df.columns:
        if latest["RSI_6"] < 30 and pre["RSI_6"] >= 30:
            signals.append("RSI6超卖反弹")
            explain.append("【RSI超卖反弹】：今日RSI6跌破30出现反弹，短期见底迹象。")
        else:
            explain.append(f"【RSI超卖反弹】：RSI6当前为{latest['RSI_6']:.1f}，未触发超卖反弹。")
    else:
        explain.append("【RSI超卖反弹】：数据不足，无法判断。")
    if "volume" in df.columns and "close" in df.columns and len(df) >= 6:
        pre_vol = df["volume"].iloc[-6:-1].mean()
        vol_up = latest["volume"] > 1.5 * pre_vol
        pct_chg = (latest["close"] - pre["close"]) / pre["close"] * 100 if pre["close"] > 0 else 0
        if vol_up and pct_chg > 2:
            signals.append("放量突破")
            explain.append("【放量突破】：今日成交量明显放大，且涨幅超过2%，主力资金有启动迹象。")
        else:
            explain.append(f"【放量突破】：今日成交量{latest['volume']}，均量{pre_vol:.0f}，{'放量' if vol_up else '未放量'}，涨幅{pct_chg:.2f}%。")
    else:
        explain.append("【放量突破】：数据不足，无法判断。")
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] >= df["close"].iloc[-20:].max():
            signals.append("20日新高")
            explain.append("【20日新高】：今日收盘价达到近20日最高。")
        else:
            explain.append(f"【20日新高】：今日收盘{latest['close']}，20日最高{df['close'].iloc[-20:].max()}，未创新高。")
    else:
        explain.append("【20日新高】：数据不足，无法判断。")
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] <= df["close"].iloc[-20:].min():
            signals.append("20日新低")
            explain.append("【20日新低】：今日收盘价达到近20日最低。")
        else:
            explain.append(f"【20日新低】：今日收盘{latest['close']}，20日最低{df['close'].iloc[-20:].min()}，未创新低。")
    else:
        explain.append("【20日新低】：数据不足，无法判断。")
    return signals, explain
