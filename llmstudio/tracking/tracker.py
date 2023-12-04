import requests
import json

# URL for the POST request
URL = 'http://localhost:8080/api/tracking/logs/'

# Headers to be sent with the request
HEADERS = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

class Tracker:
    def __init__(self, url):
        self._session = requests.Session()
        self._url = url
    
    def log(self, data: dict):
        req = self._session.post(self._url, 
                                 headers=HEADERS, 
                                 data=json.dumps(data),
                                 timeout=100)
        return req

tracker = Tracker(URL)
