import json
import requests
url = 'https://pilpal.onrender.com/chat'
url1 = 'https://hacksavvy.onrender.com/chat'
# First message
message = {'message': 'dart and dolo'}
response = requests.post(url, json=message)
response_data = response.json()
print(response_data)
print("----------------------------------------------------------------------------")
response_data_str = json.dumps(response_data)
message2 = {'message': 'What will be the side effects of ' + response_data_str+'On a person having have arrhythmia'}
response2 = requests.post(url1, json=message2)
response_data2 = response2.json()
print(response_data2)
