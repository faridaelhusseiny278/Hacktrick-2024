import requests
from LSBSteg import *
import random
from riddle_solvers import riddle_solvers
api_base_url = "http://3.70.97.142:5000"
team_id="eK9pJh6"


def init_fox(team_id):
    '''
    In this function you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will receive back the message that you can break into chuncks
      and the carrier image that you will encode the chunk in it.
    '''
    """
     This function initializes the game as a fox by sending a POST request to the 
     `/fox/start` endpoint and returning the secret message and carrier image.

     Args:
         team_id (str): The ID of the team participating in the game.

     Returns:
         tuple: A tuple containing two elements:
             - secret_message (str): The secret message to be broken into chunks.
             - carrier_image (np.array): The carrier image used to encode the message chunks.

     Raises:
         Exception: If the request fails or the response is unsuccessful.
     """
    # Endpoint URL
    url = f"{api_base_url}/fox/start"

    # Request data
    data = {"teamId": team_id}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

    # Check for successful response
    if response.status_code != 200 and response.status_code != 201:
        raise Exception(f"Request failed with status code: {response.status_code}")

    # Parse response data
    response_data = response.json()
    secret_message = response_data["msg"]
    carrier_image = np.array(response_data["carrier_image"])

    return secret_message, carrier_image


def generate_message_array(message, image_carrier: np.array):
    print(message)
    '''
    In this function you will need to create your own strategy. That includes:
        1. How you are going to split the real message into chunks
        2. Include any fake chunks
        3. Decide what 3 chunks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunk in the image carrier
    '''
    #Create variable called turns=3 and split the message by turns
    turns = 6  # Number of turns (channels) to send data
    real_chunk_size = int(np.ceil(len(message) / turns))  # Chunk size based on turns

    # Split the message into chunks
    real_message_chunks = [message[i:i + real_chunk_size] for i in range(0, len(message), real_chunk_size)]

    # Define fake message
    fake_message = "Thank you!"
    fake_chunks_size = int(np.ceil(len(fake_message) / turns))  # Chunk size based on turns
    fake_message_chunks = [fake_message[i:i + fake_chunks_size] for i in range(0, len(fake_message), fake_chunks_size)]

    fake_2_message = "you can do it"
    fake_2_chunks_size = int(np.ceil(len(fake_2_message) / turns))  # Chunk size based on turns
    fake_2_message_chunks = [fake_2_message[i:i + fake_2_chunks_size] for i in range(0, len(fake_2_message), fake_2_chunks_size)]

    # Initialize array to hold message arrays for each turn
    message_arrays = []

    for i in range(turns):
        # Determine real, fake, and empty chunks for this turn
        real_chunk = encode(np.ndarray.copy(image_carrier), real_message_chunks[i])
        fake_chunk = encode(np.ndarray.copy(image_carrier), fake_message_chunks[i])
        fake_2_chunk = encode(np.ndarray.copy(image_carrier), fake_2_message_chunks[i])  # You can modify this according to your requirement
        # Randomize the order of real, fake, and empty chunks
        chunks_order = [('R', real_chunk), ('F', fake_chunk), ('F', fake_2_chunk)]
        random.shuffle(chunks_order)
        # Append the shuffled chunks to the message array for this turn
        message_arrays.append(chunks_order)

    return message_arrays


def get_riddle(team_id, riddle_id):
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that: 
        1. Once you requested a riddle you cannot request it again per game. 
        2. Each riddle has a timeout if you did not reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle. 
    '''
    # Endpoint URL
    url = f"{api_base_url}/fox/get-riddle"

    # Request data
    data = {"teamId": team_id, "riddleId": riddle_id}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

    # Check for successful response
    if response.status_code != 200 and response.status_code != 201:
        raise Exception(f"Request failed with status code: {response.status_code}")

    # Return response data
    return response.json()['test_case']


def solve_riddle(team_id, solution):
    '''
    In this function you will solve the riddle that you have requested. 
    You will hit the API end point that submits your answer.
    Use the riddle_solvers.py to implement the logic of each riddle.
    '''
    """
    This function sends a POST request to the `/fox/solve-riddle` endpoint to submit a solution to the requested riddle.

    Args:
        team_id (str): The ID of the team participating in the game.
        solution (str): The solution to the riddle.

    Returns:
        dict: The response dictionary containing the budget information and solution status.

    Raises:
        Exception: If the request fails or the response is unsuccessful.
    """

    # Endpoint URL
    url = f"{api_base_url}/fox/solve-riddle"

    # Request data
    data = {"teamId": team_id, "solution": solution}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

    # Check for successful response
    if response.status_code != 200 and response.status_code != 201:
        raise Exception(f"Request failed with status code: {response.status_code}")

    print(response.json()["total_budget"])

    # Return response data
    return response.json()["status"]


def send_message(team_id, messages_input):
    '''
    Use this function to call the api end point to send one chunk of the message. 
    You will need to send the message (images) in each of the 3 channels along with their entities.
    Refer to the API documentation to know more about what needs to be sent in this api call.
    '''
    url = f"{api_base_url}/fox/send-message"
    messages = []
    message_entities = []
    for i in range(3):
        message_entities.append(messages_input[i][0])
        messages.append(messages_input[i][1].tolist())
    data = {
        "teamId": team_id,
        "messages": messages,
        "message_entities": message_entities
    }
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

    # Check for successful response
    if response.status_code != 200 and response.status_code != 201:
        raise Exception(f"Request failed with status code: {response.status_code}")

    # Return response text
    return response.json()["status"]


def end_fox(team_id):
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this function will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring function
      (Like failing to submit the entire message within the timelimit of the game).
    '''
    """
    This function sends a POST request to the `/fox/end-game` endpoint to conclude the game for the Fox.

    Args:
        team_id (str): The ID of the team participating in the game.

    Returns:
        str: The response text indicating the score and new high score status.

    Raises:
        Exception: If the request fails or the response is unsuccessful.
      """

    # Endpoint URL
    url = f"{api_base_url}/fox/end-game"

    # Request data
    data = {"teamId": team_id}

    # Send POST request
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

    # Check for successful response
    if response.status_code != 200 and response.status_code != 201 and response.status_code != 400:
        raise Exception(f"Request failed with status code: {response.status_code}")

    print(response.text)
    # Return response text
    return response.text


def submit_fox_attempt(team_id):
    '''
     Call this function to start playing as a fox. 
     You should submit with your own team id that was sent to you in the email.
     Remember you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve riddles 
        3. Make your own Strategy of sending the messages in the 3 channels
        4. Make your own Strategy of splitting the message into chunks
        5. Send the messages 
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be at least R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling 
    '''
    secret_message, carrier_image = init_fox(team_id)

    # Riddles
    riddles = ['problem_solving_easy', 'problem_solving_medium', 'problem_solving_hard', 'sec_hard', 'ml_medium', 'cv_hard' ]
    for i in range(len(riddles)):
        test_case = get_riddle(team_id, riddles[i])
        sol = riddle_solvers[riddles[i]](test_case)
        solve_riddle(team_id, sol)

    # Message
    message_arrays = generate_message_array(secret_message, np.ndarray.copy(carrier_image))
    for i in range(len(message_arrays)):
        send_message(team_id, message_arrays[i])

    end_fox(team_id)

if __name__ == "__main__":
    end_fox(team_id)
