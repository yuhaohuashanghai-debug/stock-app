# 📈 A股批量AI自动选股 & 智能趋势预测

基于 **Streamlit + AkShare + pandas-ta + Plotly + OpenAI**  
实现以下功能：
- 热门行业/概念板块排行
- 全市场/ETF/指数/板块池分批选股
- 自动检测信号（金叉、超卖、放量突破、新高新低）
- 可导出结果到 Excel
- AI 趋势点评（需提供 OpenAI API Key）

## 🚀 部署
1. Fork 本仓库到自己的 GitHub
2. 登录 [Streamlit Cloud](https://share.streamlit.io/)
3. 选择 **Deploy an app** → 连接 GitHub 仓库
4. 填写：
   - Repository: `yourname/stock-app`
   - Branch: `main`
   - File path: `app.py`
5. 点击 Deploy，等待 2-3 分钟即可访问。

## 🔑 注意
- AI点评功能需在页面输入 OpenAI API Key。
- 如果遇到加载慢，请耐心等待，接口有缓存机制。
