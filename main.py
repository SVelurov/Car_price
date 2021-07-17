import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Предсказание цен машин

cars = pd.read_csv('cars_new.csv', sep=',')


# Создаём словарь поле - его индекс
def create_dict(s):
    ret = {}
    for id, name in enumerate(s):
        ret.update({name: id})
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

# Запоминаем цены
prices = np.array(cars['price'], dtype=np.float)

# Запоминаем числовые параметры и нормируем
years = preprocessing.scale(cars['year'])
mileages = preprocessing.scale(cars['mileage'])
volumes = preprocessing.scale(cars['volume'])
powers = preprocessing.scale(cars['power'])

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

'''Опробованны следуюшие регресивные модели:
*linear_model.LinearRegression()
*linear_model.SGDRegressor()
*linear_model.BayesianRidge()
*linear_model.LassoLars(
*linear_model.PassiveAggressiveRegressor()

Модель BayesianRidge() показала лучший результат по метрике MAE '''

# Обучаем модель и делаем прогноз
clf = linear_model.BayesianRidge()
clf.fit(X_train, y_train_scaled)
price_car = clf.predict(X_test)
# Запоминаем модель
filename = '/content/drive/MyDrive/Colab_Notebooks/Data/price_car.sav'
pickle.dump(clf, open(filename, 'wb'))

# Подгружаеи модель
loaded_model = pickle.load(open(filename, 'rb'))
price_car = loaded_model.predict(X_test)
print(price_car)

# Восстанавливаем предсказанные цены
price_car_pred = scaler.inverse_transform(price_car)

# Проверяем качество прогноза

print('Forecast results:')
mae = mae(y_test, price_car_pred)
rms = rms(y_test, price_car_pred, squared=False)

print("mae: %.2f" % mae)
print("rms: %.2f" % rms)
