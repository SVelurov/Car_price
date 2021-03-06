Итоговый проект курса "Машинное обучение в бизнесе"

ML: keras.models, sklearn, pandas, numpy

API: Flask

Данные: база данных машин спарсенная с сайта Юла

Задача: предсказать по анализу существующих обьявлений на сайте 
стоимость машины на основании нескольких параметров. 
Регрессионная модель.

Используемые признаки:

- mark (text) "Марка машины"
- model (text) "Модель машины"
- year (number) "Год выпуска"
- mileage (number) "Пробег"
- body (text) "Форма кузова"
- kpp (text) "Тип коробки передач"
- fuel (text) "Топливо"
- volume (number) "Обьем двигателя в литрах"
- power (number) "Мощность двигателя в л.с."
- price (number) "Стоимость машины"

Преобразования признаков: OHE, Scaling

Проверен ряд ML моделей:
linear_model.LinearRegression()
linear_model.SGDRegressor()
linear_model.BayesianRidge()
linear_model.LassoLars(
linear_model.PassiveAggressiveRegressor()

Лучший результат (по mae и rms) показала модель BayesianRidge()
mae: 84934.58  (руб)
rms: 273344.80

Возможные меры по улучшению:
1. Добавить в обработчик Flask нормализацию цифровых параметров
   (год, пробег, обьем и мощность).
Для теста при обучении модели была использована нормализация (в коде оставил в комментах) 
   и точность возросла на 9%. При реализации обработчика Flask от нормализации пришлось
   отказаться, запутался в размерностях (((
2. Использовать нейронную сеть. Для теста была опробована простейшая модель 
   на 2 полносвязанных слоях (32 - 1) и точность модели оказалась не несколько процентов
   выше, чем на ML модели.
   
Модель находится в файле Main.ry

Для запуска Flask сервера используется файл Flask_server.ry

Для запуска клиента используется файл Price_request.ry

Можно менять параметры машины в запросе, модель сразу пересчитывает цену.
HTML реализации нет.

Работу сервиса можно посмотреть на Google Colab:

Модель - https://colab.research.google.com/drive/13mGqWWAFWR1zkZIYupLWmqNibqFb_3yb?usp=sharing

Flask сервер - https://colab.research.google.com/drive/1LhPFUbGL3N9j82lTW0KyBxWzXo23hSwj?usp=sharing

Клиент - https://colab.research.google.com/drive/13jvROWDYAY_F8AHAzYvlAM06jdgDn6gr?usp=sharing
