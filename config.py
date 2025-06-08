"""
OptiQuery Configuration Settings
"""

# Analysis Settings
ANALYSIS_SETTINGS = {
    "complexity_thresholds": {
        "low": 30,
        "medium": 60,
        "high": 90
    },
    "benchmark_iterations": 3,
    "max_result_size": 10000,
    "timeout_seconds": 30
}

# Security Settings
SECURITY_SETTINGS = {
    "sensitive_patterns": [
        "password",
        "credit_card",
        "ssn",
        "secret",
        "token",
        "api_key",
        "private_key"
    ],
    "blocked_keywords": [
        "DROP",
        "TRUNCATE",
        "DELETE FROM",
        "ALTER TABLE"
    ],
    "require_where_clause": True,
    "max_union_count": 3
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    "row_warning_threshold": 1000000,
    "execution_time_warning": 5.0,  # seconds
    "index_size_warning": 1000000,  # rows
    "max_join_tables": 5
}

# Data Quality Settings
DATA_QUALITY_SETTINGS = {
    "null_check_required": True,
    "type_conversion_warning": True,
    "max_string_length": 255,
    "date_format": "YYYY-MM-DD",
    "required_constraints": [
        "primary_key",
        "foreign_key",
        "not_null"
    ]
}

# Documentation Settings
DOCUMENTATION_SETTINGS = {
    "required_sections": [
        "overview",
        "purpose",
        "performance",
        "security",
        "dependencies",
        "maintenance"
    ],
    "auto_generate_diagrams": True,
    "include_examples": True
}

# Best Practices
BEST_PRACTICES = {
    "naming_conventions": {
        "table_prefix": "tbl_",
        "view_prefix": "vw_",
        "index_prefix": "idx_",
        "primary_key_suffix": "_id"
    },
    "code_style": {
        "keywords_case": "upper",
        "max_line_length": 80,
        "indent_spaces": 4
    }
}

# Error Messages
ERROR_MESSAGES = {
    "security_violation": "⚠️ Security violation detected: {details}",
    "performance_warning": "⚠️ Performance concern: {details}",
    "data_quality_issue": "⚠️ Data quality issue: {details}",
    "best_practice_violation": "ℹ️ Best practice suggestion: {details}"
}

# Query Categories
QUERY_CATEGORIES = {
    "reporting": {
        "timeout": 60,
        "max_rows": 50000,
        "allow_aggregations": True
    },
    "operational": {
        "timeout": 30,
        "max_rows": 1000,
        "require_transaction": True
    },
    "analytical": {
        "timeout": 120,
        "max_rows": 1000000,
        "allow_temp_tables": True
    }
}

# Optimization Strategies
OPTIMIZATION_STRATEGIES = {
    "index_recommendation": {
        "min_table_size": 1000,
        "column_selectivity_threshold": 0.1
    },
    "query_rewrite": {
        "enable_subquery_optimization": True,
        "enable_join_reordering": True,
        "enable_union_optimization": True
    },
    "materialization": {
        "candidate_min_executions": 100,
        "refresh_interval_minutes": 60
    }
}

# Monitoring Settings
MONITORING_SETTINGS = {
    "log_queries": True,
    "log_level": "INFO",
    "metrics_collection": {
        "execution_time": True,
        "row_count": True,
        "cpu_usage": True,
        "memory_usage": True
    },
    "alert_thresholds": {
        "execution_time_ms": 5000,
        "row_count": 1000000,
        "error_rate": 0.01
    }
} 