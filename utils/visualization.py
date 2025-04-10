# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_basic_plot(df, numeric_column: str) -> str:
    """
    Строит гистограмму по numeric_column и сохраняет график в файл.
    Возвращает путь к сохраненному файлу.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df[numeric_column], kde=True)
    plt.title(f"Распределение {numeric_column}")
    plt.xlabel(numeric_column)
    plt.ylabel("Частота")
    file_path = f"plots/{numeric_column}_hist.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(file_path)
    plt.close()
    return file_path
