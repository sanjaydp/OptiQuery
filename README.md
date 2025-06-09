# 🧠 OptiQuery: SQL Optimizer Assistant

OptiQuery is an AI-powered SQL query optimization tool designed to analyze, optimize, and improve the performance of SQL queries. Built with Streamlit and leveraging OpenAI's GPT, OptiQuery provides detailed suggestions for improving SQL query performance, best practices, and security, along with benchmarking, execution time analysis, and other advanced features.

## ✨ Features

* 🚀 SQL Query Optimization: Optimizes your SQL queries for better performance, readability, and security.

* 🗣️ Natural Language to SQL: Converts natural language queries into SQL statements using OpenAI's GPT-3.

* 🔍 Advanced SQL Analysis: Analyzes queries for syntax errors, performance issues, data quality impacts, and security risks.

* 📊 Benchmarking: Benchmarks SQL queries against sample data and shows the performance improvements.

* 📝 Comprehensive Query Reports: Generates detailed reports that explain the query's performance, security, complexity, and optimization suggestions.

* 💾 Database Support: Supports SQLite (default) and PostgreSQL databases.

* 📋 Copy SQL to Clipboard: Copy optimized SQL queries with a single click for easy use.

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/sanjaydp/OptiQuery.git
cd OptiQuery
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Set up the environment variables by creating a .env file. Add your OpenAI API key (optional but required for AI-powered features):

```plaintext
OPENAI_API_KEY=your_openai_api_key
```

## 🚀 Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

Once the app is running, you will be able to:

* 📤 Upload an SQL file or directly paste a SQL query to analyze.

* ⚡ Get optimized SQL queries and performance improvement suggestions.

* 🤖 Convert natural language queries into SQL using GPT-3.

* ⏱️ Benchmark your SQL queries and view execution times.

* 📜 Explore query history and stored results.

## 🔑 Key Components

### 1. 🎯 SQL Query Optimization:
* Optimizes SQL queries to improve their performance and ensures that they follow best practices.

* Suggests indexing recommendations and possible query rewrites.

* Checks for performance bottlenecks and suggests ways to mitigate them.

### 2. 🗣️ Natural Language to SQL:
* Converts natural language queries into SQL queries using OpenAI's GPT model.

* Makes querying databases more user-friendly for those unfamiliar with SQL syntax.

### 3. 🛡️ Security and Data Quality Analysis:
* Analyzes SQL queries for security risks such as SQL injection vulnerabilities.

* Checks for potential data quality issues, such as missing NULL checks or data truncation.

### 4. 📊 Benchmarking:
* Measures the execution time of SQL queries under different conditions.

* Runs multiple benchmarks and calculates average performance metrics.

### 5. 📜 Query History:
* Stores and displays previous queries with performance and execution time details.

* Allows users to compare original vs optimized query performance.

## ⚙️ Configuration

* 🗄️ SQLite: The app defaults to SQLite for local database use. You can either upload an existing SQLite database or use the built-in sample database.

* 🐘 PostgreSQL: Optionally, you can switch to PostgreSQL if it's available by setting the proper connection details in the sidebar.

* 🤖 OpenAI: Set your OpenAI API key to enable GPT-powered features (natural language to SQL and query optimization).

## 💡 Example Use Case

* 🔄 SQL Query Optimization: Upload an SQL query file or paste a query, and the app will analyze it for performance improvements. The app will provide detailed suggestions, such as adding indexes, optimizing joins, and removing redundant operations.

* 💬 Natural Language to SQL: Enter a natural language query like "Show me all orders from customers in the North region after January 2024 with total over $1000", and the app will generate the corresponding SQL query for you.

* 📈 Benchmarking: Run your SQL queries against a sample database and see how the execution time improves after optimization. View the number of rows processed and the estimated performance gains.

## 🔮 Future Enhancements

* 🔌 Support for More Databases: Extend support to other database systems like MySQL, MSSQL, etc.

* 🎯 Automated Index Creation: Automatically suggest and create indexes based on query patterns.

* 📊 Advanced Query Plan Visualizations: Provide better visualization of the query execution plan for more complex queries.

## 🤝 Contributing

Feel free to open issues or submit pull requests if you find any bugs or want to contribute to new features!

1. 🍴 Fork the repository
2. 🌿 Create a new branch for your feature (git checkout -b feature-name)
3. ✍️ Make your changes
4. 💾 Commit your changes (git commit -am 'Add new feature')
5. ⬆️ Push to the branch (git push origin feature-name)
6. 🎯 Create a new Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

* 🤖 OpenAI GPT-3 for generating SQL queries from natural language.
* 🌟 Streamlit for creating the web app interface.
* 🗄️ SQLite for the lightweight database.
* 🐘 PostgreSQL for enterprise-level database support.

---

This README provides an overview of the OptiQuery app and should help users understand how to get started with it. Feel free to customize it further to match your specific needs! ✨
