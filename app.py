import streamlit as st
import akshare as ak
import pandas as pd
import pandas_ta as ta
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="A股热门板块热力榜+批量选股", layout="wide")
st.title("🔥 A股热门板块动态排行 + 批量智能选股信号")

# ========== 1. 热门板块榜单与可视化 ==========
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    board_type = st.radio("榜单类型", options=["概念板块", "行业板块"], horizontal=True)
with col2:
    refresh_btn = st.button("手动刷新榜单", key="refresh_board")
with col3:
    st.markdown(f"**数据更新时间：{datetime.now().strftime('%H:%M:%S')}**")

# 拉取数据
@st.cache_data(ttl=300)
def get_hot_board_df(board_type="concept", topn=20):
    if board_type == "concept":
        df = ak.stock_board_concept_name_ths()
    elif board_type == "industry":
        df = ak.stock_board_industry_name_ths()
    else:
        return pd.DataFrame()
    df = df.sort_values("涨跌幅", ascending=False).head(topn)
    return df

board_key = "concept" if board_type == "概念板块" else "industry"
if refresh_btn:
    st.cache_data.clear()
hot_df = get_hot_board_df(board_key)

if not hot_df.empty:
    st.subheader(f"🔥 今日{board_type}涨幅热力榜 TOP20")
    # 可视化
    fig = px.bar(
        hot_df,
        x="涨跌幅",
        y="板块名称",
        orientation='h',
        text="涨跌幅",
        color="涨跌幅",
        color_continuous_scale="RdYlGn",
        height=600,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(hot_df[["板块名称", "涨跌幅"]], use_container_width=True)
else:
    st.warning("未能获取板块排行数据，请稍后重试")

# ========== 2. 点选板块加载成分股池 ==========
if not hot_df.empty:
    selected_board = st.selectbox(
        "点击热门板块自动加载成分股：",
        hot_df["板块名称"].tolist(),
        key="board_select"
    )
    if selected_board:
        @st.cache_data(ttl=600)
        def get_board_codes(board_type, board_name):
            if board_type == "concept":
                df = ak.stock_board_concept_cons_ths(symbol=board_name)
            else:
                df = ak.stock_board_industry_cons_ths(symbol=board_name)
            return df["代码"].tolist()
        codes = get_board_codes(board_key, selected_board)
        st.success(f"板块【{selected_board}】成分股已加载，共{len(codes)}只")
    else:
        codes = []
else:
    codes = []

# ========== 3. 批量自动选股信号检测 ==========

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
    except Exception as e:
        pass
    return df

def signal_with_explain(df):
    explain = []
    signals = []
    latest = df.iloc[-1]
    pre = df.iloc[-2] if len(df) >= 2 else latest
    # 1. 均线金叉
    if "SMA_5" in df.columns and "SMA_10" in df.columns:
        if pre["SMA_5"] < pre["SMA_10"] and latest["SMA_5"] > latest["SMA_10"]:
            signals.append("5日均线金叉10日均线")
            explain.append("【均线金叉】：5日均线上穿10日均线（金叉），多头信号。")
        else:
            explain.append(f"【均线金叉】：5日均线({latest['SMA_5']:.2f}) {'>' if latest['SMA_5']>latest['SMA_10'] else '<='} 10日均线({latest['SMA_10']:.2f})，未发生金叉。")
    else:
        explain.append("【均线金叉】：数据不足，无法判断。")
    # 2. MACD金叉
    if "MACD" in df.columns and "MACDs" in df.columns:
        if pre["MACD"] < pre["MACDs"] and latest["MACD"] > latest["MACDs"]:
            signals.append("MACD金叉")
            explain.append("【MACD金叉】：MACD线上穿信号线，金叉出现，多头信号。")
        else:
            explain.append(f"【MACD金叉】：MACD({latest['MACD']:.3f}) {'>' if latest['MACD']>latest['MACDs'] else '<='} 信号线({latest['MACDs']:.3f})，未发生金叉。")
    else:
        explain.append("【MACD金叉】：数据不足，无法判断。")
    # 3. RSI超卖反弹
    if "RSI_6" in df.columns:
        if latest["RSI_6"] < 30 and pre["RSI_6"] >= 30:
            signals.append("RSI6超卖反弹")
            explain.append("【RSI超卖反弹】：今日RSI6跌破30出现反弹，短期见底迹象。")
        else:
            explain.append(f"【RSI超卖反弹】：RSI6当前为{latest['RSI_6']:.1f}，未触发超卖反弹。")
    else:
        explain.append("【RSI超卖反弹】：数据不足，无法判断。")
    # 4. 放量突破
    if "volume" in df.columns and "close" in df.columns and len(df) >= 6:
        pre_vol = df["volume"].iloc[-6:-1].mean()
        vol_up = latest["volume"] > 1.5 * pre_vol
        pct_chg = (latest["close"] - pre["close"]) / pre["close"] * 100 if pre["close"] > 0 else 0
        if vol_up and pct_chg > 2:
            signals.append("放量突破")
            explain.append("【放量突破】：今日成交量放大，涨幅超2%，有主力资金启动迹象。")
        else:
            explain.append(f"【放量突破】：今日成交量{latest['volume']}，均量{pre_vol:.0f}，{'放量' if vol_up else '未放量'}，涨幅{pct_chg:.2f}%。")
    else:
        explain.append("【放量突破】：数据不足，无法判断。")
    # 5. 20日新高
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] >= df["close"].iloc[-20:].max():
            signals.append("20日新高")
            explain.append("【20日新高】：今日收盘价达近20日最高。")
        else:
            explain.append(f"【20日新高】：今日收盘{latest['close']}，20日最高{df['close'].iloc[-20:].max()}，未创新高。")
    else:
        explain.append("【20日新高】：数据不足，无法判断。")
    # 6. 20日新低
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] <= df["close"].iloc[-20:].min():
            signals.append("20日新低")
            explain.append("【20日新低】：今日收盘价达近20日最低。")
        else:
            explain.append(f"【20日新低】：今日收盘{latest['close']}，20日最低{df['close'].iloc[-20:].min()}，未创新低。")
    else:
        explain.append("【20日新低】：数据不足，无法判断。")
    return signals, explain

# 选股池信号检测（防止过多卡顿，建议<100只为佳）
if codes:
    st.info(f"开始批量检测板块成分股信号（共{len(codes)}只标的，建议每次不超100只）")
    start_date = st.date_input("选择起始日期", value=pd.to_datetime("2024-01-01"), key="pick_start")
    btn = st.button("一键批量自动选股", key="btn_pick")
    if btn:
        result_table = []
        for i, code in enumerate(codes):
            df = fetch_ak_data(code, start_date)
            if df.empty or len(df) < 25:
                continue
            df = calc_indicators(df)
            signals, explain = signal_with_explain(df)
            result_table.append({
                "代码": code,
                "信号": "、".join(signals) if signals else "无明显信号",
                "明细解释": "\n".join(explain)
            })
            # 仅前10实时打印到页面
            if i < 10:
                st.markdown(f"#### 【{code}】选股信号：{'、'.join(signals) if signals else '无明显信号'}")
                with st.expander("信号检测明细（点击展开）", expanded=False):
                    for line in explain:
                        st.write(line)
        selected = [r for r in result_table if "无明显信号" not in r["信号"]]
        if selected:
            st.subheader("✅ 入选标的与信号（可全部导出）")
            df_sel = pd.DataFrame(selected)
            st.dataframe(df_sel[["代码","信号"]])
            st.download_button(
                "导出全部明细为Excel",
                data=pd.DataFrame(result_table).to_excel(index=False),
                file_name="选股明细.xlsx"
            )
        else:
            st.warning("暂无标的触发选股信号，可调整策略或换池。")
else:
    st.info("请先在上方选择热门板块，并加载成分股池后再运行批量信号检测。")
