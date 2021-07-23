import requests

url = "http://127.0.0.1:5000/predict"

payload = {"mark": "kia", "model": "cerato", "year": "2010", "mileage": "300000", \
           "body": "седан", "kpp": "автомат", "fuel": "бензин", "volume": "3.0", \
           "power": "220"}

predicted_price = requests.post(url, json=payload)

if predicted_price:
    print('Ориентировочная стоимость машины:')
else:
    print('An error has occurred.')

price = predicted_price.text.replace('"1":', '')[1:-13]
print(f'{price} руб.')