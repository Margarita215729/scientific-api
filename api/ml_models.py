# api/ml_models.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

router = APIRouter()

class MLRequest(BaseModel):
    file_id: str  # для примера будем использовать CSV с данными
    target_column: str

class MLResponse(BaseModel):
    accuracy: float
    f1_score: float
    roc_auc: float
    best_params: dict

@router.post("/train", response_model=MLResponse)
async def train_model(request: MLRequest):
    """
    Загружает данные (для примера из локального файла, можно объединить с функцией скачивания), 
    делит на выборки, обучает классификатор, проводит GridSearch и возвращает метрики.
    """
    try:
        # Для демонстрации читаем локальный CSV файл – в реальном проекте скачивание можно объединить с drive_utils
        df = pd.read_csv("sample_data.csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки данных: {str(e)}")
    
    if request.target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Указанный target_column отсутствует в данных.")
    
    # Простая предобработка: удаляем строки с пропусками
    df = df.dropna()
    
    X = df.drop(columns=[request.target_column])
    y = df[request.target_column]
    
    # Для простоты оставляем только числовые признаки
    X = X.select_dtypes(include=["number"])
    
    if X.empty:
        raise HTTPException(status_code=400, detail="Нет числовых признаков для обучения.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Обучаем RandomForest с GridSearch
    clf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10]
    }
    gs = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy")
    gs.fit(X_train, y_train)
    
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Рассчитываем метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    try:
        roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])
    except Exception:
        roc = 0.0  # если невозможен расчет ROC-AUC (например, бинарная классификация не подходит)
    
    return MLResponse(
        accuracy=acc,
        f1_score=f1,
        roc_auc=roc,
        best_params=gs.best_params_
    )
