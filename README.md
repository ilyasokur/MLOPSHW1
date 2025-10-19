# Kaggle ML Inference Service

Сервис для **инференса** модели внутри Docker.
Вход: `./input/test.csv`.  
Выход: `./output/sample_submission.csv` + артефакты:
- `feature_importances_top5.json` — топ-5 фичей.
- `prediction_density.png` — плотность/распределение предсказаний.

## 1) Подготовка

1. Склонировать репозиторий.
2. Распаковать model.pkl.zip в том же каталоге (github не хотел его принимать в изначальном виде так как > 100mb)
3. Создание директорий mkdir input
4. Положить test.csv в каталог input
5. Build Docker:
    ```bash
    docker build --no-cache -t kaggle-ml-service:latest .
6. Run Docker:
    ```bash
    docker run --rm -v $(pwd)/input:/mnt/input -v $(pwd)/output:/mnt/output kaggle-ml-service:latest
## 2) Результат

1. В каталоге output - появиться ./output/sample_submission.csv + артефакты