"""
Enterprise-grade features for OptiQuery
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Any
import sqlparse
import re

def add_enterprise_sidebar():
    """Add enterprise-grade settings to the sidebar"""
    with st.sidebar:
        st.markdown("## ⚙️ Analysis Settings")
        
        # Query Category Selection
        query_category = st.selectbox(
            "Query Category",
            ["Reporting", "Operational", "Analytical"],
            help="Select the type of query you're optimizing"
        )
        
        # Optimization Strategy
        optimization_strategy = st.selectbox(
            "Optimization Strategy",
            ["Balanced", "Performance", "Readability", "Security"],
            help="Choose optimization priority"
        )
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            advanced_settings = {
                "benchmark_iterations": st.slider("Benchmark Iterations", 1, 10, 3),
                "security_level": st.select_slider(
                    "Security Level",
                    ["Low", "Medium", "High"],
                    value="Medium"
                ),
                "enable_ai_suggestions": st.checkbox("Enable AI Suggestions", True),
                "show_query_metrics": st.checkbox("Show Query Metrics", True)
            }
        
        return query_category, optimization_strategy, advanced_settings

def analyze_enterprise_features(query: str) -> Dict[str, Any]:
    """Perform enterprise-grade analysis of the query"""
    results = {
        "security": analyze_security(query),
        "performance": analyze_performance(query),
        "best_practices": analyze_best_practices(query),
        "maintainability": analyze_maintainability(query)
    }
    return results

def analyze_security(query: str) -> Dict[str, Any]:
    """Analyze query for security concerns"""
    issues = []
    recommendations = []
    risk_level = "Low"
    
    # Check for sensitive patterns
    sensitive_patterns = ["password", "credit", "ssn", "secret", "key"]
    for pattern in sensitive_patterns:
        if pattern in query.lower():
            issues.append(f"⚠️ Potential sensitive data exposure: {pattern}")
            recommendations.append("Consider data masking or encryption")
            risk_level = "High"
    
    # Check for dangerous operations
    dangerous_ops = ["DROP", "TRUNCATE", "DELETE", "UPDATE"]
    for op in dangerous_ops:
        if op in query.upper():
            issues.append(f"⚠️ Dangerous operation detected: {op}")
            recommendations.append("Implement proper access controls")
            risk_level = "High"
    
    return {
        "issues": issues,
        "recommendations": recommendations,
        "risk_level": risk_level
    }

def analyze_performance(query: str) -> Dict[str, Any]:
    """Analyze query for performance optimizations"""
    issues = []
    recommendations = []
    
    # Check for SELECT *
    if "SELECT *" in query.upper():
        issues.append("⚠️ Using SELECT * may impact performance")
        recommendations.append("Specify required columns explicitly")
    
    # Check for proper indexing hints
    if "WHERE" in query.upper() and "INDEX" not in query.upper():
        recommendations.append("Consider adding appropriate indexes")
    
    # Check for DISTINCT usage
    if "DISTINCT" in query.upper():
        issues.append("⚠️ DISTINCT operation can be expensive")
        recommendations.append("Consider using GROUP BY instead")
    
    return {
        "issues": issues,
        "recommendations": recommendations
    }

def analyze_best_practices(query: str) -> Dict[str, Any]:
    """Analyze query against SQL best practices"""
    issues = []
    recommendations = []
    
    # Check for proper aliasing
    if "JOIN" in query.upper() and " AS " not in query.upper():
        recommendations.append("Use meaningful table aliases")
    
    # Check for proper case consistency
    keywords = ["SELECT", "FROM", "WHERE", "JOIN"]
    if not all(kw in query.upper() for kw in keywords if kw.lower() in query.lower()):
        recommendations.append("Use consistent casing for SQL keywords")
    
    return {
        "issues": issues,
        "recommendations": recommendations
    }

def analyze_maintainability(query: str) -> Dict[str, Any]:
    """Analyze query maintainability"""
    issues = []
    recommendations = []
    
    # Check query length
    if len(query.split('\n')) > 50:
        issues.append("⚠️ Query is very long")
        recommendations.append("Consider breaking down into smaller queries")
    
    # Check for comments
    if not any(line.strip().startswith('--') for line in query.split('\n')):
        recommendations.append("Add descriptive comments")
    
    return {
        "issues": issues,
        "recommendations": recommendations
    }

def generate_enterprise_report(query: str, analysis_results: Dict) -> str:
    """Generate comprehensive enterprise analysis report"""
    report = f"""# Enterprise SQL Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Query Overview
```sql
{query}
```

## Security Analysis
Risk Level: {analysis_results['security']['risk_level']}

Issues:
{chr(10).join(f"- {issue}" for issue in analysis_results['security']['issues'])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['security']['recommendations'])}

## Performance Analysis
Issues:
{chr(10).join(f"- {issue}" for issue in analysis_results['performance']['issues'])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['performance']['recommendations'])}

## Best Practices
Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['best_practices']['recommendations'])}

## Maintainability
Issues:
{chr(10).join(f"- {issue}" for issue in analysis_results['maintainability']['issues'])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['maintainability']['recommendations'])}
"""
    return report

def display_enterprise_analysis(query: str, analysis_results: Dict):
    """Display enterprise analysis results in the UI"""
    st.markdown("## 🏢 Enterprise Analysis")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Security Risk", analysis_results['security']['risk_level'])
    with col2:
        st.metric("Performance Issues", len(analysis_results['performance']['issues']))
    with col3:
        st.metric("Best Practice Suggestions", len(analysis_results['best_practices']['recommendations']))
    with col4:
        st.metric("Maintainability Issues", len(analysis_results['maintainability']['issues']))
    
    # Detailed analysis sections
    with st.expander("🔒 Security Analysis", expanded=True):
        st.markdown(f"**Risk Level: {analysis_results['security']['risk_level']}**")
        for issue in analysis_results['security']['issues']:
            st.warning(issue)
        for rec in analysis_results['security']['recommendations']:
            st.info(rec)
    
    with st.expander("⚡ Performance Analysis", expanded=True):
        for issue in analysis_results['performance']['issues']:
            st.warning(issue)
        for rec in analysis_results['performance']['recommendations']:
            st.info(rec)
    
    with st.expander("✨ Best Practices", expanded=True):
        for rec in analysis_results['best_practices']['recommendations']:
            st.info(rec)
    
    with st.expander("🔧 Maintainability", expanded=True):
        for issue in analysis_results['maintainability']['issues']:
            st.warning(issue)
        for rec in analysis_results['maintainability']['recommendations']:
            st.info(rec)
    
    # Export options
    st.markdown("### 📊 Export Analysis")
    report = generate_enterprise_report(query, analysis_results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Report (MD)",
            report,
            file_name="enterprise_analysis.md",
            mime="text/markdown"
        )
    with col2:
        st.download_button(
            "📊 Download Analysis Data (JSON)",
            json.dumps(analysis_results, indent=2),
            file_name="enterprise_analysis.json",
            mime="application/json"
        ) 