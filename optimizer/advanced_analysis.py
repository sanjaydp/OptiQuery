"""
Advanced SQL Analysis Module
"""
import sqlparse
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from config import (
    ANALYSIS_SETTINGS,
    SECURITY_SETTINGS,
    PERFORMANCE_SETTINGS,
    DATA_QUALITY_SETTINGS,
    BEST_PRACTICES
)

@dataclass
class QueryMetrics:
    """Data class for storing query metrics"""
    complexity_score: float
    table_count: int
    join_count: int
    where_conditions: int
    aggregations: int
    subqueries: int
    estimated_rows: int
    execution_time: float

class QueryAnalyzer:
    """Advanced SQL Query Analyzer"""
    
    def __init__(self, query: str, schema_info: Dict = None):
        self.query = query
        self.schema_info = schema_info
        self.parsed_query = sqlparse.parse(query)[0]
        self.tokens = [token for token in self.parsed_query.tokens if not token.is_whitespace]
        
    def analyze_complexity(self) -> Dict[str, Any]:
        """Analyze query complexity using multiple factors"""
        score = 0
        factors = {}
        
        # Analyze basic structure
        factors['table_count'] = len(re.findall(r'FROM\s+\w+|JOIN\s+\w+', self.query, re.I))
        factors['join_count'] = len(re.findall(r'JOIN', self.query, re.I))
        factors['where_conditions'] = len(re.findall(r'AND|OR', self.query, re.I))
        factors['aggregations'] = len(re.findall(r'COUNT|SUM|AVG|MAX|MIN', self.query, re.I))
        factors['subqueries'] = len(re.findall(r'\(SELECT', self.query, re.I))
        
        # Calculate weighted score
        score += factors['table_count'] * 10
        score += factors['join_count'] * 15
        score += factors['where_conditions'] * 5
        score += factors['aggregations'] * 8
        score += factors['subqueries'] * 20
        
        # Normalize score
        score = min(100, score)
        
        return {
            'score': score,
            'level': self._get_complexity_level(score),
            'factors': factors,
            'recommendations': self._generate_complexity_recommendations(factors)
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze query performance characteristics"""
        issues = []
        recommendations = []
        
        # Check for full table scans
        if not re.search(r'WHERE|JOIN.*ON', self.query, re.I):
            issues.append("Full table scan detected")
            recommendations.append("Add appropriate WHERE clause or JOIN conditions")
        
        # Check for proper indexing opportunities
        join_conditions = re.findall(r'ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)', self.query, re.I)
        for cond in join_conditions:
            recommendations.append(f"Consider indexing {cond[0]} and {cond[1]}")
        
        # Check for inefficient patterns
        if 'SELECT *' in self.query.upper():
            issues.append("Selecting all columns")
            recommendations.append("Specify required columns explicitly")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'estimated_impact': self._estimate_performance_impact()
        }
    
    def analyze_security(self) -> Dict[str, Any]:
        """Analyze query for security concerns"""
        vulnerabilities = []
        risk_level = "low"
        
        # Check for dangerous operations
        for keyword in SECURITY_SETTINGS['blocked_keywords']:
            if keyword in self.query.upper():
                vulnerabilities.append(f"Dangerous operation detected: {keyword}")
                risk_level = "high"
        
        # Check for sensitive data exposure
        for pattern in SECURITY_SETTINGS['sensitive_patterns']:
            if pattern in self.query.lower():
                vulnerabilities.append(f"Potential sensitive data exposure: {pattern}")
                risk_level = "medium"
        
        return {
            'vulnerabilities': vulnerabilities,
            'risk_level': risk_level,
            'recommendations': self._generate_security_recommendations(vulnerabilities)
        }
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze query for data quality impacts"""
        issues = []
        
        # Check NULL handling
        if DATA_QUALITY_SETTINGS['null_check_required']:
            if not re.search(r'IS\s+NULL|IS\s+NOT\s+NULL', self.query, re.I):
                issues.append("No NULL value handling detected")
        
        # Check data type conversions
        if DATA_QUALITY_SETTINGS['type_conversion_warning']:
            if re.search(r'CAST|CONVERT', self.query, re.I):
                issues.append("Data type conversion detected - verify precision/scale")
        
        return {
            'issues': issues,
            'recommendations': self._generate_quality_recommendations(issues),
            'impact_level': self._assess_quality_impact(issues)
        }
    
    def generate_optimization_plan(self) -> Dict[str, Any]:
        """Generate comprehensive query optimization plan"""
        complexity = self.analyze_complexity()
        performance = self.analyze_performance()
        security = self.analyze_security()
        quality = self.analyze_data_quality()
        
        optimizations = []
        priority = "low"
        
        # Determine optimization priority
        if complexity['score'] > 70 or security['risk_level'] == "high":
            priority = "high"
        elif complexity['score'] > 40 or len(performance['issues']) > 2:
            priority = "medium"
        
        # Collect all recommendations
        optimizations.extend(performance['recommendations'])
        optimizations.extend(self._generate_index_recommendations())
        optimizations.extend(self._generate_rewrite_suggestions())
        
        return {
            'priority': priority,
            'optimizations': optimizations,
            'complexity_score': complexity['score'],
            'security_risk': security['risk_level'],
            'estimated_improvement': self._estimate_optimization_impact()
        }
    
    def _get_complexity_level(self, score: float) -> str:
        """Determine complexity level based on score"""
        thresholds = ANALYSIS_SETTINGS['complexity_thresholds']
        if score <= thresholds['low']:
            return "Low"
        elif score <= thresholds['medium']:
            return "Medium"
        else:
            return "High"
    
    def _generate_complexity_recommendations(self, factors: Dict) -> List[str]:
        """Generate recommendations based on complexity factors"""
        recommendations = []
        
        if factors['join_count'] > PERFORMANCE_SETTINGS['max_join_tables']:
            recommendations.append("Consider breaking down query into smaller parts")
        
        if factors['subqueries'] > 2:
            recommendations.append("Consider replacing subqueries with JOINs")
        
        return recommendations
    
    def _generate_security_recommendations(self, vulnerabilities: List[str]) -> List[str]:
        """Generate security-focused recommendations"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if "sensitive data" in vuln:
                recommendations.append("Implement column-level encryption")
            elif "dangerous operation" in vuln:
                recommendations.append("Use parameterized queries")
        
        return recommendations
    
    def _generate_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        for issue in issues:
            if "NULL" in issue:
                recommendations.append("Add explicit NULL handling")
            elif "conversion" in issue:
                recommendations.append("Validate data type conversion logic")
        
        return recommendations
    
    def _generate_index_recommendations(self) -> List[str]:
        """Generate index recommendations"""
        recommendations = []
        
        # Analyze WHERE clauses
        where_columns = re.findall(r'WHERE\s+(\w+\.\w+)', self.query, re.I)
        for col in where_columns:
            recommendations.append(f"Create index on {col}")
        
        # Analyze JOIN conditions
        join_columns = re.findall(r'ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)', self.query, re.I)
        for left, right in join_columns:
            recommendations.append(f"Create composite index on {left}, {right}")
        
        return recommendations
    
    def _generate_rewrite_suggestions(self) -> List[str]:
        """Generate query rewrite suggestions"""
        suggestions = []
        
        # Check for DISTINCT usage
        if 'DISTINCT' in self.query.upper():
            suggestions.append("Consider replacing DISTINCT with GROUP BY")
        
        # Check for complex subqueries
        if len(re.findall(r'\(SELECT', self.query, re.I)) > 1:
            suggestions.append("Consider using CTEs for better readability")
        
        return suggestions
    
    def _estimate_performance_impact(self) -> str:
        """Estimate performance impact of current query structure"""
        impact = 0
        
        # Factor in various metrics
        if 'SELECT *' in self.query.upper():
            impact += 20
        if not re.search(r'WHERE', self.query, re.I):
            impact += 30
        if len(re.findall(r'JOIN', self.query, re.I)) > 3:
            impact += 25
        
        return f"{min(impact, 100)}% potential performance improvement"
    
    def _estimate_optimization_impact(self) -> Dict[str, Any]:
        """Estimate impact of suggested optimizations"""
        return {
            'performance_improvement': self._estimate_performance_impact(),
            'complexity_reduction': "30-40%",
            'maintenance_impact': "Medium"
        }
    
    def _assess_quality_impact(self, issues: List[str]) -> str:
        """Assess the impact on data quality"""
        if len(issues) > 3:
            return "High"
        elif len(issues) > 1:
            return "Medium"
        return "Low" 