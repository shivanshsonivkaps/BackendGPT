import requests
import json

# Define the URL
# url = "http://127.0.0.1:5000/api"
url = "http://localhost:5000/api"

# Define the payload (JSON body)
payload = {
    "question": "Different views of an architetcture"
}

# Convert payload to JSON format
headers = {
    "Content-Type": "application/json"
}

# Send POST request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Print the response
print(response.json())

