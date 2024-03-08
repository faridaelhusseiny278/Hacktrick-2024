# Add the necessary imports here
import pandas as pd
from SteganoGAN.utils import *
from DES import *
from app import *
from test2 import *
import joblib
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
df=pd.read_csv('E:\dell_hacktrick\MlMediumTrainingData.csv')
def solve_cv_easy(test_case: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    return []


def solve_cv_medium(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    return remove_patch(base_image = input[0], patch_image = input[1])


def solve_cv_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """

    return answer_image(input[0], input[1])

#image = Image.open("dogcat.jpg")
#img_list = list(image.getdata())
#width, height = image.size
#
## Reshape the 2D list to a 3D list
#img_3d_list = [img_list[i * width:(i + 1) * width] for i in range(height)]
#print(solve_cv_hard(('How many dogs?', img_3d_list)))

def solve_ml_easy(data: pd.DataFrame) -> list:

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)


    data.fillna(data.mean(), inplace=True)


    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = data['visits'].quantile(0.25)
    Q3 = data['visits'].quantile(0.75)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers
    outliers = data[(data['visits'] < lower_bound) | (data['visits'] > upper_bound)]
    # Remove outliers
    data_no_outliers = data[~((data['visits'] < lower_bound) | (data['visits'] > upper_bound))]

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_no_outliers)

    # Define a function to prepare data for LSTM model
    def prepare_data(data, n_steps):
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix > len(data) - 1:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    # Define the number of time steps
    n_steps = 30

    # Prepare data for LSTM model
    X, y = prepare_data(scaled_data, n_steps)

    # Split data into training and validation sets (using the entire dataset for training)
    train_size = len(X)
    X_train, y_train = X[:train_size], y[:train_size]

    loaded_model = load_model('E:\dell_hacktrick\my_model.h5')

    forecast = []
    current_batch = X[-1].reshape((1, n_steps, X_train.shape[2]))
    for i in range(50):
        current_pred = loaded_model.predict(current_batch)[0]
        forecast.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # Inverse scaling to get the actual forecasted values
    forecast = scaler.inverse_transform(forecast)
    final_forecast = np.floor(forecast)
    final_list = final_forecast.tolist()
    print(f"ml_easy = {final_list}")
    return final_list

def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    dbscan = DBSCAN(eps=2, min_samples=5)
    df['cluster'] = dbscan.fit_predict(df[['x_','y_']])
    new_point = [input]

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(df[['x_', 'y_']])
    distances, indices = neigh.kneighbors(new_point)

    if distances[0][0] <= dbscan.eps and df.iloc[indices[0][0]]["cluster"] != -1:
        predicted_cluster = df.iloc[indices[0][0]]["cluster"]
    else:
        predicted_cluster = -1

    return int(predicted_cluster)

#print(solve_ml_medium([1.0, 1.0]))
def solve_sec_medium(input: list) -> str:
    """
        This function takes a torch.Tensor as input and returns a string as output.

        Parameters:
        input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

        Returns:
        str: A string representing the decoded message from the image.
    """
    img_array = np.array(input, dtype=np.uint8)
    # Create a PIL Image from the numpy array
    img = Image.fromarray(img_array)
    # Convert the image to a tensor
    img_tensor = torch.tensor(img.getdata(), dtype=torch.float32).view(img.size[1], img.size[0], 3)
    # Normalize the pixel values to the range [0, 1]
    img_tensor /= 255.0
    # Transpose the dimensions to match PyTorch format (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    # Add a batch dimension (unsqueeze)
    img_tensor = img_tensor.unsqueeze(0)
    decoded_data = decode(img_tensor)
    return decoded_data

def solve_sec_hard(input:tuple)->str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """
    key = input[0]
    pt = input[1]

    return func(key, pt)

def solve_problem_solving_easy(input: tuple) -> list:
    """This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    n = len(input[0])
    x= input[1]
    listed = list(input[0])
    listed.sort()
    count_arr=[]
    i=0
    arr= []
    while i<n:
        count = 1
        arr.append(listed[i])
        while i+1<n and listed[i]==listed[i+1]:
            i+=1
            count+=1
        count_arr.append(count)
        i+=1
    combined = list(zip(count_arr, arr))
    
    # Sort the list of tuples based on the integer values
    combined.sort(key=lambda x: x[0], reverse=True)
    
    # Separate the sorted tuples back into separate arrays
    sorted_arr_int, sorted_arr_str = zip(*combined)
    res = []
    for i in range(x):
      res.append(sorted_arr_str[i])
    return res



def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    decoded_string = ""
    index = 0
    while index < len(input):
        if input[index].isdigit():
            num_start = index
            while index < len(input) and input[index].isdigit():
                index += 1
            num = int(input[num_start:index])
            count = 1
            j = index + 1
            while j < len(input) and count != 0:
                if input[j] == '[':
                    count += 1
                elif input[j] == ']':
                    count -= 1
                j += 1
            substr = solve_problem_solving_medium(input[index + 1:j - 1])
            decoded_string += substr * min(num, 10 ** 5 // len(substr))
            index = j
        else:
            decoded_string += input[index]
            index += 1
    return decoded_string[:10 ** 5]


def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    if input[0] == 1 and input[1] == 1:
        return 1
    Length_n = [0]
    for t in range(input[1] - 1):
        Length_n.append(1)

    for i in range(input[0] - 1):
        Length_o = Length_n
        Length_n = [1]
        for j in range(1, input[1]):
            Length_n.append(Length_n[j - 1] + Length_o[j])
    return Length_n[-1]


riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    'sec_medium_stegano': solve_sec_medium,
    'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}


#tpl = (['mina', 'mina', 'mazen'], 1)
#solve_problem_solving_easy(tpl)
#
#img = Image.open('encoded.png')
## Get pixel dimensions
#width, height = img.size
#
## Create an empty list to store pixel data
#pixel_data = []
#
## Iterate through each row (y-coordinate) and column (x-coordinate)
#for y in range(height):
#    row_data = []
#    for x in range(width):
#        # Get pixel value (RGB tuple)
#        pixel = img.getpixel((x, y))
#        row_data.append(pixel)
#    pixel_data.append(row_data)



#print(decode(img_tensor))

#print(solve_sec_medium(pixel_data))
#print(solve_sec_hard(('266200199BBCDFF1', '0123456789ABCDEF')))

#print(solve_sec_medium(pixel_data))