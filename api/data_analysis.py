# api/data_analysis.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from io import BytesIO
import pandas as pd
from utils.drive_utils import download_file_from_drive
from utils.report_generator import generate_markdown_report
from utils.visualization import create_basic_plot

router = APIRouter()

class AnalysisResponse(BaseModel):
    columns: list
    num_rows: int
    dtypes: dict
    description: dict
    report: str

@router.get("/analyze")
async def analyze_file(file_id: str):
    """
    Скачивает CSV-файл с Google Drive по file_id, анализирует его и генерирует отчет.
    """
    try:
        file_stream = download_file_from_drive(file_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки файла: {str(e)}")
    
    try:
        df = pd.read_csv(file_stream)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {str(e)}")
    
    # Предварительный анализ
    analysis = {
        "columns": list(df.columns),
        "num_rows": df.shape[0],
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "description": df.describe(include="all").to_dict()
    }
    
    # Генерация визуализации (например, гистограмма первой числовой колонки, если есть)
    plot_path = None
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        plot_path = create_basic_plot(df, numeric_cols[0])
    
    # Генерация markdown-отчета
    report = generate_markdown_report(analysis, plot_path)
    
    analysis["report"] = report
    
    return analysis
