Итоговый проект курса "Машинное обучение в бизнесе"

ML: keras.models, sklearn, pandas, numpy
API: flask
Данные: база данных машин с сайта Юла спарсенная для другого проекта

Задача: предсказать по анализу существующих обьявлений на сайте стоимость машины на основании нескольких параметров. 
Регрессионная модель

Используемые признаки:

- mark (text)-
- model (text)
- year (number)
- mileage (number)
- body (text)
- kpp (text)
- fuel (text)
- volume (number)
- power (number)
- price (number)

Преобразования признаков: OHE, Scaling

Проверен ряд ML моделей:
linear_model.LinearRegression()
linear_model.SGDRegressor()
linear_model.BayesianRidge()
linear_model.LassoLars(
linear_model.PassiveAggressiveRegressor()

Лучший результат (по mae и rms) показала модель BayesianRidge()
mae: 84934.46  (руб)
rms: 273344.80

Однако, простейшая нейронная сеть на 3 полносвязанных слоях показала
еще лучше результат:
mae: 69526.21   (руб)
rms: 215836.02

В проекте будет использована нейронная сеть.

### Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/SVelurov/Car_price.git
$ cd Car_price
$ docker build -t SVelurov/Car_price .
```

### Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -p 8181:8181 -v C:\Users\4ma\PycharmProjects\Car_price:/app/app/models SVelurov/Car_price.git
```

### Переходим на localhost:8181
