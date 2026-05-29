# Credit Risk Scoring API

Real-time credit risk assessment system built with FastAPI and XGBoost.

**Live Demo:** https://credit-risk-api-t4h1.onrender.com  
**API Docs:** https://credit-risk-api-t4h1.onrender.com/docs  
**Dashboard:** https://credit-risk-api-t4h1.onrender.com/dashboard

---

## What It Does
Accepts 10 borrower features and returns:
- Credit risk score (150–850 scale, mirrors CIBIL logic)
- Default probability (%)
- Risk category: LOW / MEDIUM / HIGH
- Recommendation: APPROVE / MANUAL REVIEW / DECLINE

## Tech Stack
- **Model:** XGBoost trained on 150,000 borrower records (0.81 AUC)
- **API:** FastAPI + Uvicorn
- **Deployment:** Render.com

## Built By
Kayalvizhi S | M.S. FinTech, SRM Institute