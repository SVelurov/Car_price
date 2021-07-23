import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as RMS

# Загружаем данные из файла
cars = pd.read_csv("cars_new.csv", sep=',')
print(cars[:5])


# Создаём словарь поле - его индекс
def create_dict(s):
    ret = {}
    for _id, name in enumerate(s):
        ret.update({name: _id})
    return ret

# Функция преобразования в one hot encoding
def to_ohe(value, d):
    arr = [0] * len(d)
    arr[d[value]] = 1
    return arr

# Создаём словари по всем текстовым колонкам
marks_dict = create_dict(set(cars['mark']))
models_dict = create_dict(set(cars['model']))
bodies_dict = create_dict(set(cars['body']))
kpps_dict = create_dict(set(cars['kpp']))
fuels_dict = create_dict(set(cars['fuel']))
dict_car = []
dict_car.append(marks_dict)
dict_car.append(models_dict)
dict_car.append(bodies_dict)
dict_car.append(kpps_dict)
dict_car.append(fuels_dict)

# Запоминаем словарь с параметрами
with open('dict_car.pkl', 'wb') as f:
    pickle.dump(dict_car, f)

# Запоминаем цены
prices = np.array(cars['price'], dtype=np.float)

'''# Запоминаем числовые параметры и нормируем
scaler = StandardScaler()
years = scaler.fit_transform([cars['year']]).flatten()
with open('scaler_years.pkl', 'wb') as f:
    pickle.dump(scaler, f)
mileages = scaler.fit_transform([cars['mileage']]).flatten()
with open('scaler_mileages.pkl', 'wb') as f:
    pickle.dump(scaler, f)
volumes = scaler.fit_transform([cars['volume']]).flatten()
with open('scaler_volumes.pkl', 'wb') as f:
    pickle.dump(scaler, f)
powers = scaler.fit_transform([cars['power']]).flatten()
with open('scaler_powers.pkl', 'wb') as f:
    pickle.dump(scaler, f)'''

# Запоминаем числовые параметры, без нормирования
years = list(cars['year'])
mileages = list(cars['mileage'])
volumes = list(cars['volume'])
powers = list(cars['power'])

# Создаём пустую обучающую выборку
X = []
y = []

# Проходим по всем машинам и в y_train добавляем цену
for _id, car in enumerate(np.array(cars)):
    y.append(prices[_id])

    # Объединяем все параметры (категорийные в виде ohe, числовые напрямую)
    x_tr = to_ohe(car[0], marks_dict) + \
           to_ohe(car[1], models_dict) + \
           to_ohe(car[5], bodies_dict) + \
           to_ohe(car[6], kpps_dict) + \
           to_ohe(car[7], fuels_dict) + \
           [years[_id]] + \
           [mileages[_id]] + \
           [volumes[_id]] + \
           [powers[_id]]

    # Добавляем текущую строку в общий x_train
    X.append(x_tr)

# Превращаем лист в numpy.array
X = np.array(X, dtype=np.float)
y = np.array(y, dtype=np.float)

# Делим всю выборку на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализуем y_train
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
with open('scaler_car.pkl', 'wb') as f:
    pickle.dump(scaler, f)

'''Опробованны следуюшие модели:
*linear_model.LinearRegression()
*linear_model.SGDRegressor()
*linear_model.BayesianRidge()
*linear_model.LassoLars()
*linear_model.PassiveAggressiveRegressor()

Выбрана BayesianRidge() показавшая лучший результат по метрике MAE'''

# Обучаем модель и делаем прогноз
clf = linear_model.BayesianRidge()
clf.fit(X_train, y_train_scaled)
price_car = clf.predict(X_test)

# Запоминаем модель
filename = "price_car.sav"
pickle.dump(clf, open(filename, 'wb'))

# Восстанавливаем предсказанные цены
price_car_pred = scaler.inverse_transform(price_car)

# Проверяем качество прогноза
print('Forecast results:')
mae = MAE(y_test, price_car_pred)
rms = RMS(y_test, price_car_pred, squared=False)

print("mae: %.2f" % (mae))
print("rms: %.2f" % (rms))