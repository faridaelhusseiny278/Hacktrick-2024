import numpy as np
from LSBSteg import decode
import requests
from sklearn.preprocessing import  StandardScaler
from keras.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = load_model('model.h5')
from sklearn.impute import KNNImputer


knnimputer = KNNImputer(n_neighbors=5) # global KNN Imputer
api_base_url = "http://3.70.97.142:5000"
team_id="eK9pJh6"
scaler = StandardScaler()


def init_eagle(team_id):

    # Endpoint URL
    url = f"{api_base_url}/eagle/start"

    # Request data
    data = {"teamId": team_id}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

    # Check for successful response
    if response.status_code != 200 and response.status_code != 201:
        raise Exception(f"Request failed with status code: {response.status_code}")
    
    # Parse response data
    response_data = response.json() 
    spec1 = np.array(response_data['footprint']['1'])
    spec2 = np.array(response_data['footprint']['2'])
    spec3 = np.array(response_data['footprint']['3'])
    inputs = np.array([spec1, spec2,spec3])
   
    return inputs


def skip_msg(team_id):
    url = f"{api_base_url}/eagle/skip-message"
    # Request data
    data = {"teamId": team_id}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    if response.status_code != 200 and response.status_code != 201  and response.status_code != 400:
        raise Exception(f"Request failed with status code: {response.status_code}")
    
    if response.text == "End of message reached":
            end_eagle(team_id)
    else:
            response_data = response.json()
            inputs = get_mydf(response_data)
            channel = Predict_Channel(inputs)
            if channel:
                request_msg(team_id, channel)
            else:
                skip_msg(team_id)
    

def request_msg(team_id, channel_id):
    url = f"{api_base_url}/eagle/request-message"
    # Request data
    data = {"teamId": team_id, "channelId":channel_id}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    if response.status_code != 200 and response.status_code != 201:
        raise Exception(f"Request failed with status code: {response.status_code}")
    
    # Parse response data
    response_data = response.json()
    encodedMsg = response_data['encodedMsg']
    encodedMsg = np.array(encodedMsg)
    decoded_Msg = decode(encodedMsg.copy())
    submit_msg(team_id, decoded_Msg)


def submit_msg(team_id, decoded_msg):
    url = f"{api_base_url}/eagle/submit-message"
    # Request data

    data = {"teamId": team_id,"decodedMsg":decoded_msg}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    if response.status_code != 200 and response.status_code != 201  and response.status_code != 400:
        raise Exception(f"Request failed with status code: {response.status_code}")
    
    if response.text == "End of message reached":
            end_eagle(team_id)
    else:
            response_data = response.json()
            inputs = get_mydf(response_data)
            channel = Predict_Channel(inputs)
            if channel:
                request_msg(team_id, channel)
            else:
                skip_msg(team_id)
    
         
    

    
def get_mydf(response_data):
    spec1 = np.array(response_data['nextFootprint']['1'])
    spec2 = np.array(response_data['nextFootprint']['2'])
    spec3 = np.array(response_data['nextFootprint']['3'])
    inputs = np.array([spec1, spec2,spec3])
    return inputs



def Predict_Channel(inputs):
    Reals={}
    for index, input in enumerate(inputs):
        index+=1
        input[np.isinf(input)] = np.nan
        #nan_indices = np.isnan(input)
        #non_nan_values = input[~nan_indices]
        
        #random_values = np.random.choice(non_nan_values, size=np.sum(nan_indices))
        input = knnimputer.fit_transform(input)

        
        input = scaler.fit_transform(input)
        input = np.expand_dims(input, axis=-1)
        input = np.expand_dims(input, axis=(0, 3))

        prediction = model.predict(input, verbose=0)

        if (np.argmax(prediction)) == 1:
            Reals[index] = prediction
            
    if Reals:
        min_key = min(Reals, key=lambda k: np.min(Reals[k]))
        return min_key
    else:
        return None
        
    
def submit_eagle_attempt(team_id):

    inputs = init_eagle(team_id)
    channel = Predict_Channel(inputs)
    if channel:
        request_msg(team_id, channel)
    else:
        skip_msg(team_id)


def end_eagle(team_id):
    # Endpoint URL
    url = f"{api_base_url}/eagle/end-game"

    # Request data
    data = {"teamId": team_id}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

    # Check for successful response
    if response.status_code != 200 and response.status_code != 201 and response.status_code != 400:
        raise Exception(f"Request failed with status code: {response.status_code}")
    
    print(response.text)        

submit_eagle_attempt(team_id)