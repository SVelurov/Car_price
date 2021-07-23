from flask import Flask, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Создаем сервис для обработки запросов к модели

# Загружаем обученную модель
def start_model():
    model = pickle.load(open("price_car.sav", 'rb'))
    return model

# Предварительная обработка данных для загрузки в модель
def pre_processing(mark, model, year, mileage, body, kpp, fuel, volume, power):
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

    # Загружаем словари по всем текстовым колонкам
    dict_car = pickle.load(open('dict_car.pkl', 'rb'))

    marks_dict = dict_car[0]
    models_dict = dict_car[1]
    bodies_dict = dict_car[2]
    kpps_dict = dict_car[3]
    fuels_dict = dict_car[4]

    '''# Восстанавливаем числовые параметры и нормируем
    scaler = pickle.load(open(path+'scaler_years.pkl','rb'))
    years = scaler.transform(np.array(year).reshape(1, -1))
    scaler = pickle.load(open(path+'scaler_mileages.pkl','rb'))
    mileages = scaler.transform(np.array(mileage).reshape(1, -1))
    scaler_volumes = pickle.load(open(path+'scaler_volumes.pkl','rb'))
    volumes = scaler.transform(np.array(volume).reshape(1, -1))
    scaler_powers = pickle.load(open(path+'scaler_powers.pkl','rb'))
    powers = scaler.transform(np.array(power).reshape(1, -1))'''

    # Создаём пустую выборку и заполняем данными
    X = []
    x_tr = to_ohe(mark, marks_dict) + \
           to_ohe(model, models_dict) + \
           to_ohe(body, bodies_dict) + \
           to_ohe(kpp, kpps_dict) + \
           to_ohe(fuel, fuels_dict) + \
           [float(year.strip().strip("'"))] + \
           [float(mileage.strip().strip("'"))] + \
           [float(volume.strip().strip("'"))] + \
           [float(power.strip().strip("'"))]

    # Добавляем строку в обучающую выборку
    X.append(x_tr)

    # Превращаем лист в numpy.array
    X = np.array(X, dtype=np.float)

    return X

# Обработчики и запуск Flask
app = Flask(__name__)

# Конфигурируем Flask сервер
@app.route('/predict', methods=['GET', 'POST'])
def get_predict():
    if request.method == "POST":
        request_json = request.get_json()
        mark = request_json["mark"]
        model = request_json["model"]
        year = request_json["year"]
        mileage = request_json["mileage"]
        body = request_json["body"]
        kpp = request_json["kpp"]
        fuel = request_json["fuel"]
        volume = request_json["volume"]
        power = request_json["power"]

        X = pre_processing(mark, model, year, mileage, body, kpp, fuel, volume, power)

        # Запускаем модель
        model = start_model()
        price_car = model.predict(X)

        # Восстанавливаем значение цены машины
        scaler = pickle.load(open('scaler_car.pkl', 'rb'))
        price_car_pred = scaler.inverse_transform(price_car).flatten()

        return dict(enumerate(price_car_pred.flatten(), 1))

# Запускаем модель и Flask сервер
if __name__ == '__main__':
    print("Загружаем Keras модель и запускаем Flask сервер...")
    start_model()
    app.run()