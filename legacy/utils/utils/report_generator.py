# utils/report_generator.py
def generate_markdown_report(analysis: dict, plot_path: str = None) -> str:
    """
    Генерирует markdown-отчет на основе анализа данных и (опционально) добавляет ссылку на график.
    """
    report = "# Отчет по анализу данных\n\n"
    report += "## Основные характеристики\n"
    report += f"- Количество строк: **{analysis.get('num_rows', 'N/A')}**\n"
    report += f"- Столбцы: {', '.join(analysis.get('columns', []))}\n\n"
    report += "## Типы данных\n"
    for col, dtype in analysis.get("dtypes", {}).items():
        report += f"- **{col}**: {dtype}\n"
    report += "\n## Описательная статистика\n"
    for stat, values in analysis.get("description", {}).items():
        report += f"### {stat}\n"
        for col, value in values.items():
            report += f"- {col}: {value}\n"
        report += "\n"
    if plot_path:
        report += "## Визуализация\n"
        report += f"![График]({plot_path})\n\n"
    report += "## Заключение\n"
    report += "На основе проведенного анализа можно рекомендовать дальнейшие шаги по обработке и обучению моделей.\n"
    return report
