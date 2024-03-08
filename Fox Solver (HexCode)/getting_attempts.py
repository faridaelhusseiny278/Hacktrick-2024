import requests

api_base_url = "http://13.53.169.72:5000/attempts/student"
team_id="eK9pJh6"

# Request data
data = {"teamId": team_id}

# Send POST request
response = requests.post(api_base_url, json=data, headers={"Content-Type": "application/json"})

# Check for successful response
if response.status_code != 200 and response.status_code != 201:
    raise Exception(f"Request failed with status code: {response.status_code}")
# Parse response data
response_data = response.json()
print(response_data)