import csv
import pandas as pd
from datetime import datetime
import math
import numpy as np
from typing import Dict, List
from tensorflow.keras.utils import to_categorical

def read_endpoints(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return [line[0] for line in reader]

def save_endpoints(path, endpoints):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f) 
        for i in range(len(endpoints)):
            writer.writerow([i, endpoints[i]])

def read_csv_into_dictionary(path):
    dict_from_csv = {}
    with open(path, mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0]:rows[1] for rows in reader}
    return dict_from_csv

def read_csv_into_array(path):
    array = np.genfromtxt(path, delimiter=',')
    return array

def read_raw_user_operations(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return [[user, endpoint, datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f'), datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f').date()] for user, endpoint, timestamp in reader]

def read_anonymize_user_operations_into_dataframe(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = [[user, endpoint, datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f'), datetime.strptime(date, '%Y-%m-%d')] for user, endpoint, timestamp, date in reader]
        df = pd.DataFrame(data, columns=['User','Endpoint','Timestamp', 'Date'])
        return df

def anonimize_operations(ops_df:pd.DataFrame):
    users = ops_df.User.unique()
    users.sort()
    # change ops_df['User'] to be the index of the user in users
    ops_df['User'] = ops_df['User'].apply(lambda x: users.tolist().index(x))

def get_endpoints_from_operations(ops_df:pd.DataFrame) -> pd.DataFrame:
    endpoints = ops_df.Endpoint.unique()
    endpoints.sort()
    return endpoints

def save_dictionary_to_csv(path, dictionary):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f) 
        for key,val in dictionary.items():
            writer.writerow([key, val.split(',')])

def save_dictionary_array_file(path, dictionary):
    np.savetxt(path, [value.round(6, value) for key,value in dictionary.items()], delimiter=',')

def save_fixed_operations_dataframe(path, df:pd.DataFrame):
    df.to_csv(path, index=False)

def add_constant_endpoints(endpoints:List):
    endpoints.insert(0, '|unk|')
    endpoints.insert(0, '|eos|')
    return endpoints

def create_unique_word_dict(text:List) -> Dict:
    # Getting all the unique words from our text and sorting them alphabetically
    words = list(set(text))
    words.sort()
    # Creating the dictionary for the unique words
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({
            word: i
        })
    return unique_word_dict 

def one_hot_encode(word, dictionary):
    oh = [0 for i in range(len(dictionary))]
    oh[dictionary[word]] = 1
    return oh

def get_operations_dataframe_from_list(operations):
    df = pd.DataFrame(operations, columns=['User','Endpoint','Timestamp', 'Date'])
    return df

def split_operations_dataframe_per_user_and_date(df: pd.DataFrame):
    return [v.sort_values('Timestamp') for k,v in df.groupby(['User','Date'])]

def get_sequences_from_operations(df_list:List[pd.DataFrame], width:int, get_shorter_sequences:bool = False):
    if get_shorter_sequences:
        # walk through each list of operations and make sequence of MAX width = width
        sequences = [[df[i:min(i+width, len(df))] for i in range(len(df))] 
        for df in df_list]
    else:
        # walk through each list of operations and make sequence of width = width, as far as possible
        sequences = [[df[i:i+width] for i in range(len(df)) if i+width <= len(df)] 
            for df in df_list]
    return flatten(sequences)

def flatten(td_list:List[List[pd.DataFrame]]):
    return [sequence[i] for sequence in td_list for i in range(len(sequence))]

def get_focus_context_dataframes_from_sequences(sequences:List[pd.DataFrame]):
    # make vectors [a_i, a_{i+j}] from sequences, where a_i = focus word, a_{i+j} = context word

    # first make lists for each focus word
    focus_vectors = flatten([[ (i,df[i:]) for i in range(len(df))] for df in sequences])

    # now make one vector for each context word per focus_vector
    vectors = flatten([ [ [df[i:i+1], df[i+j:i+j+1]] for j in range(len(df)) if j > 0 and i+j+1 <= len(df)] for (i, df) in focus_vectors])
    
    return vectors

def get_arrays_from_dataframes(focus_vectors:List[pd.DataFrame]):
    return ([item[0].Endpoint.tolist()[0] for item in focus_vectors],
        [item[1].Endpoint.tolist()[0] for item in focus_vectors])

def get_sequence__arrays_and_vocabulary_from_operations(user_operations, width):
    operations_df = get_operations_dataframe_from_list(user_operations)
    operations_by_user_and_date_df = split_operations_dataframe_per_user_and_date(operations_df)
    sequences = get_sequences_from_operations(operations_by_user_and_date_df, width)
    sequence_array = np.array([sequence.Endpoint for sequence in sequences])
    endpoints = get_endpoints_from_operations(operations_df)
    return sequence_array, endpoints

def get_sequence_one_hot_arrays(sequence_array, endpoints):
    sequence_index_array = [flatten([np.where(endpoints == item)[0] for item in sequence]) for sequence in sequence_array]
    one_hot = to_categorical(sequence_index_array)
    return one_hot

def get_sequence_index_arrays(sequence_array, endpoints):
    sequence_index_array = [flatten([np.where(endpoints == item)[0] for item in sequence]) for sequence in sequence_array]
    return sequence_index_array

def separate_input_output_from_sequences(sequence_array):
    input = np.array([i[:-1] for i in sequence_array])
    output = np.array([i for i in sequence_array])
    return input, output


# operations_df = get_operations_dataframe_from_list(read_raw_user_operations('raw_user_operations.csv'))
# anonimize_operations(operations_df)
# save_fixed_operations_dataframe('user_operations_anonymized.csv', operations_df)
# endpoints = get_endpoints_from_operations(operations_df)
# operations_by_user_and_date_df = split_operations_dataframe_per_user_and_date(operations_df)
# sequences = get_sequences_from_operations(operations_by_user_and_date_df, 2)
# focus_dfs = get_focus_context_dataframes_from_sequences(sequences)
# x,y = get_arrays_from_dataframes(focus_dfs)
