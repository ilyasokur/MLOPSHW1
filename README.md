# Kaggle ML Inference Service

Сервис для **инференса** модели внутри Docker.
Вход: `./input/test.csv`.  
Выход: `./output/sample_submission.csv` + артефакты:
- `feature_importances_top5.json` — топ-5 важностей фичей.
- `prediction_density.png` — плотность/распределение предсказаний.

## 1) Подготовка

1. Склонировать репозиторий.
2. Build Docker - docker build --no-cache -t kaggle-ml-service:latest .
3. Run Docker - docker run --rm -v $(pwd)/input:/mnt/input -v $(pwd)/output:/mnt/output kaggle-ml-service:latest

## 1) Результат

1. В каталоге output - появиться ./output/sample_submission.csv + артефакты