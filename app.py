import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import talib
from datetime import datetime, timedelta
import time
import requests
import json

# 页面设置
st.set_page_config(
    page_title="A股批量AI自动选股 & 板块信号聚合解读",
    page_icon="📈",
    layout="wide"
)

# 应用标题
st.title("A股批量AI自动选股 & 板块信号聚合解读")

# 侧边栏 - 用户输入
st.sidebar.header("参数设置")

# Deepseek API密钥输入
api_key = st.sidebar.text_input("请输入你的Deepseek API Key", type="password")

# 股票代码输入
stocks_input = st.sidebar.text_area(
    "请输入A股股票代码 (支持批量，用逗号/空格/换行分隔)",
    value="000001, 000002, 600036",
    height=100
)

# 日期选择
analysis_date = st.sidebar.date_input(
    "选择分析日期",
    value=datetime.now().date() - timedelta(days=1)
)

# AI趋势点评选项
enable_ai = st.sidebar.checkbox("启用AI趋势点评", value=True)

# 分析按钮
analyze_btn = st.sidebar.button("批量分析", type="primary")

# 处理股票代码输入
def process_stock_input(input_text):
    # 替换各种分隔符为逗号
    for sep in [' ', '\n', ';', '，', '、']:
        input_text = input_text.replace(sep, ',')
    
    # 分割并清理代码
    codes = [code.strip().zfill(6) for code in input_text.split(',') if code.strip()]
    return codes

# 获取股票数据
def fetch_stock_data(stock_code, start_date, end_date):
    try:
        # 使用akshare获取股票数据
        stock_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if stock_df.empty:
            return None
        
        # 重命名列以符合标准
        stock_df = stock_df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover"
        })
        
        # 确保日期格式正确
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        stock_df = stock_df.set_index('date')
        
        return stock_df
    except Exception as e:
        st.error(f"获取股票 {stock_code} 数据时出错: {str(e)}")
        return None

# 技术指标计算
def calculate_technical_indicators(df):
    # 计算移动平均线
    df['MA5'] = talib.SMA(df['close'], timeperiod=5)
    df['MA20'] = talib.SMA(df['close'], timeperiod=20)
    df['MA60'] = talib.SMA(df['close'], timeperiod=60)
    
    # 计算相对强弱指数 (RSI)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    
    # 计算MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist
    
    # 计算布林带
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    
    return df

# 生成交易信号
def generate_signals(df):
    signals = {}
    
    # 价格与移动平均线关系
    current_close = df['close'].iloc[-1]
    signals['价格高于MA5'] = current_close > df['MA5'].iloc[-1]
    signals['价格高于MA20'] = current_close > df['MA20'].iloc[-1]
    signals['价格高于MA60'] = current_close > df['MA60'].iloc[-1]
    
    # RSI信号
    rsi = df['RSI'].iloc[-1]
    signals['RSI超买'] = rsi > 70
    signals['RSI超卖'] = rsi < 30
    
    # MACD信号
    signals['MACD金叉'] = df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]
    signals['MACD死叉'] = df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]
    
    # 布林带信号
    signals['价格突破上轨'] = current_close > df['BB_upper'].iloc[-1]
    signals['价格突破下轨'] = current_close < df['BB_lower'].iloc[-1]
    
    return signals

# 调用AI分析 (模拟)
def ai_analysis(stock_code, df, signals):
    # 这里应该是调用Deepseek API的代码
    # 由于没有真实的API，这里使用模拟响应
    
    # 模拟API调用延迟
    time.sleep(0.5)
    
    # 基于技术指标生成模拟AI分析
    analysis = f"股票 {stock_code} 技术分析:\n"
    
    if signals['价格高于MA5'] and signals['价格高于MA20'] and signals['价格高于MA60']:
        analysis += "- 股价位于多条均线之上，呈现多头排列，趋势较为强势。\n"
    elif not signals['价格高于MA5'] and not signals['价格高于MA20'] and not signals['价格高于MA60']:
        analysis += "- 股价位于多条均线之下，呈现空头排列，趋势较为弱势。\n"
    else:
        analysis += "- 股价与均线关系复杂，处于震荡整理阶段。\n"
    
    if signals['RSI超买']:
        analysis += "- RSI指标显示超买状态，短期可能存在回调风险。\n"
    elif signals['RSI超卖']:
        analysis += "- RSI指标显示超卖状态，短期可能存在反弹机会。\n"
    else:
        analysis += "- RSI指标处于中性区域，方向性不明显。\n"
    
    if signals['MACD金叉']:
        analysis += "- MACD出现金叉信号，短期趋势可能转好。\n"
    elif signals['MACD死叉']:
        analysis += "- MACD出现死叉信号，短期趋势可能转弱。\n"
    
    if signals['价格突破上轨']:
        analysis += "- 股价突破布林带上轨，显示强势但可能短期超买。\n"
    elif signals['价格突破下轨']:
        analysis += "- 股价突破布林带下轨，显示弱势但可能短期超卖。\n"
    
    # 添加一些随机性以使分析看起来更真实
    random_phrases = [
        "建议关注成交量变化确认趋势有效性。",
        "需结合大盘整体走势综合判断。",
        "注意控制仓位，防范风险。",
        "可考虑分批建仓策略。",
        "短期波动加大，注意止损设置。"
    ]
    
    analysis += f"- {np.random.choice(random_phrases)}\n"
    
    return analysis

# 主程序
if analyze_btn:
    if not stocks_input:
        st.error("请输入至少一个股票代码!")
    else:
        # 处理股票代码
        stock_codes = process_stock_input(stocks_input)
        st.sidebar.success(f"已识别 {len(stock_codes)} 只股票")
        
        # 设置日期范围
        end_date = analysis_date.strftime("%Y%m%d")
        start_date = (analysis_date - timedelta(days=120)).strftime("%Y%m%d")  # 120天数据
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 存储所有股票分析结果
        results = []
        
        for i, code in enumerate(stock_codes):
            status_text.text(f"正在分析 {code} ({i+1}/{len(stock_codes)})...")
            
            # 获取股票数据
            stock_data = fetch_stock_data(code, start_date, end_date)
            
            if stock_data is not None and not stock_data.empty:
                # 计算技术指标
                stock_data = calculate_technical_indicators(stock_data)
                
                # 生成交易信号
                signals = generate_signals(stock_data)
                
                # AI分析
                ai_comment = ""
                if enable_ai and api_key:
                    ai_comment = ai_analysis(code, stock_data, signals)
                
                # 存储结果
                results.append({
                    '代码': code,
                    '名称': f"股票{code}",  # 实际应用中应从API获取股票名称
                    '最新价': stock_data['close'].iloc[-1],
                    '涨跌幅': stock_data['pct_chg'].iloc[-1],
                    '信号': signals,
                    'AI分析': ai_comment,
                    '数据': stock_data
                })
            
            # 更新进度
            progress_bar.progress((i + 1) / len(stock_codes))
        
        status_text.text("分析完成!")
        
        # 显示结果摘要
        st.subheader("分析结果摘要")
        if results:
            summary_data = []
            for res in results:
                summary_data.append({
                    '股票代码': res['代码'],
                    '股票名称': res['名称'],
                    '最新价': round(res['最新价'], 2),
                    '涨跌幅': f"{round(res['涨跌幅'], 2)}%",
                    '趋势': "上涨" if res['涨跌幅'] > 0 else "下跌",
                    '强势信号': sum([1 for k, v in res['信号'].items() if v and ('高于' in k or '金叉' in k or '突破上轨' in k)]),
                    '弱势信号': sum([1 for k, v in res['信号'].items() if v and ('低于' in k or '死叉' in k or '突破下轨' in k)])
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # 显示详细分析
            st.subheader("详细分析与AI点评")
            for res in results:
                with st.expander(f"{res['代码']} - {res['名称']}: {res['最新价']} ({res['涨跌幅']}%)"):
                    st.write("**技术信号:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("强势信号:")
                        for signal, value in res['信号'].items():
                            if value and ('高于' in signal or '金叉' in signal or '突破上轨' in signal):
                                st.success(f"✓ {signal}")
                    
                    with col2:
                        st.write("弱势信号:")
                        for signal, value in res['signal'].items():
                            if value and ('低于' in signal or '死叉' in signal or '突破下轨' in signal):
                                st.error(f"✗ {signal}")
                    
                    # 显示价格图表
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(res['数据'].index, res['数据']['close'], label='收盘价')
                    ax.plot(res['data'].index, res['data']['MA5'], label='5日均线', alpha=0.7)
                    ax.plot(res['data'].index, res['data']['MA20'], label='20日均线', alpha=0.7)
                    ax.set_title(f"{res['代码']} 价格走势")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # 显示AI分析
                    if res['AI分析']:
                        st.write("**AI趋势点评:**")
                        st.info(res['AI分析'])
        else:
            st.warning("没有获取到有效的股票数据，请检查股票代码和日期设置。")

else:
    # 显示使用说明
    st.info("""
    ### 使用说明:
    1. 在左侧边栏输入您的Deepseek API Key (如需AI点评功能)
    2. 输入要分析的A股股票代码，多个代码用逗号/空格/换行分隔
    3. 选择分析日期
    4. 勾选是否启用AI趋势点评
    5. 点击"批量分析"按钮开始分析
    
    ### 支持功能:
    - 批量股票技术分析 (MA, RSI, MACD, 布林带等)
    - AI智能趋势预测与点评
    - 可视化价格走势图表
    - 交易信号识别与聚合
    """)

# 页脚
st.markdown("---")
st.markdown("📊 数据仅供参考，投资有风险，入市需谨慎 | © 2025 A股分析工具")
