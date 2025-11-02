# Kaggle ML Inference Service

## 1) Подготовка

1. Склонировать репозиторий.
2. Распаковать model.pkl.zip в том же каталоге (github не хотел его принимать в изначальном виде так как > 100mb)
3. Запуск - docker compose up --build -d

## 2) Ход проверки

1. Открываем kafka-ui - http://localhost:8080
2. В топик transactions можно положить вот такое сообщение:
{
	"transaction_id": "fraud-rob-001",
	"transaction_time": "2019-02-07 22:40:00",
	"merch": "fraud_Roberts, Ryan and Smith",
	"cat_id": "personal_care",
	"amount": 50.12,
	"name_1": "Carolyn",
	"name_2": "Thomas",
	"gender": "F",
	"street": "755 Solis Isle Suite 075",
	"one_city": "New Memphis",
	"us_state": "IL",
	"post_code": "62266",
	"lat": 38.4857,
	"lon": -89.6816,
	"population_city": 254,
	"jobs": "Magazine journalist",
	"merchant_lat": 38.173432,
	"merchant_lon": -90.656565,
	"target": 1
}

3. Далее идем в ui - http://localhost:8501
4. нажимаем на кнопку "Посмотреть результаты" (либо switch автообновление)