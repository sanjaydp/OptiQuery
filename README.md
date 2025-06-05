# 🧠 OptiQuery - AI-Powered SQL Optimizer Assistant

**OptiQuery** is an AI-based SQL optimization assistant built with Streamlit + OpenAI.  
It helps data engineers:
- Detect SQL anti-patterns
- Suggest performance improvements using GPT
- Visualize before/after diffs
- Execute SQL on SQLite databases
- Export markdown reports

### 🚀 Features
- Paste or upload `.sql` files
- Auto-analysis + OpenAI optimization
- Inline explanation + diff
- Test queries on uploaded SQLite DBs
- Downloadable .md reports

### 🧰 Tech Stack
- Python + Streamlit
- OpenAI API (GPT-3.5)
- sqlparse, sqlite3
- Optional deployment on Streamlit Cloud

---

### ⚙️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
