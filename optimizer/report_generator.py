def generate_report(original_query, issues, optimized_query, explanation):
    report = "### OptiQuery Optimization Report\n\n"
    report += "**Original Query:**\n```sql\n" + original_query + "\n```\n\n"
    report += "**Identified Issues:**\n"
    for issue in issues:
        report += f"- {issue}\n"
    report += "\n**Optimized Query:**\n```sql\n" + optimized_query + "\n```\n\n"
    report += "**Optimization Explanation:**\n" + explanation + "\n"
    return report
