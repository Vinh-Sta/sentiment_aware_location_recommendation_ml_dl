import requests
import json

data = {
  "text": "I stayed at the hotel for three nights with my family. The room was clean and spacious, the staff were extremely helpful and friendly, and the location was perfect, right in the city center with easy access to restaurants and attractions. I would definitely come back again!"
}


url =  "http://127.0.0.1:8888/predict/"

data = json.dumps(data)

response = requests.post(url,data)
print(response.json())


