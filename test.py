import requests


RAPIDAPI_KEY = "b04875a529msh8ac449ea5c8338ap120c6fjsn4f2de00309e0"


url = "https://jsearch.p.rapidapi.com/search-v2"


headers = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "jsearch.p.rapidapi.com"
}


params = {
    "query": "developer jobs in chicago",
    "num_pages": "1",
    "country": "us",
    "date_posted": "all"
}


response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=60
)


print("Status:", response.status_code)

print(response.text[:1000])