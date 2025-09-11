
from openai import OpenAI
import os
import time

# 初始化 OpenAI 客户端（确保环境变量 OPENAI_API_KEY 已设置）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_by_gpt(stock_code, last_row):
    # 构造 prompt
    prompt = f"""
    请你作为一名资深投资分析师，分析股票「{stock_code}」的最新技术面情况。
    当前价格：{last_row.get('close', 'N/A')}
    MACD：{last_row.get('MACD', 'N/A')}，信号线：{last_row.get('MACD_signal', 'N/A')}
    RSI：{last_row.get('RSI', 'N/A')}

    请判断当前是否适合买入、卖出或观望，并用简洁语言给出理由（包括技术指标支撑）。
    """

    for _ in range(3):  # 最多重试3次
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一个资深A股技术分析师"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)
            time.sleep(2)
    return f"❌ GPT 分析失败：{err_msg}"
