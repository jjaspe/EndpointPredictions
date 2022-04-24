import numpy as np
from utils import get_endpoints_from_operations, get_sequence__arrays_and_vocabulary_from_operations, get_sequence_index_arrays, read_anonymize_user_operations_into_dataframe, separate_input_output_from_sequences

# Usually used for concating weight matrices
# e.g W_{hh} and W_{hx} 
def hor_stack(a1, a2):
    return np.hstack((a1, a2))

# Usually used for concating input matrices
# e.g W_{hh} and W_{hx} 
def ver_stack(a1, a2):
    return np.vstack((a1, a2))

# with stacking
def hidden_t_1(h_prev_t, x_t, W_hh, W_hx, b_h):
    z = np.matmul(hor_stack(W_hh, W_hx), ver_stack(h_prev_t, x_t)) + b_h
    return sigmoid(z)

# without stacking
def hidden_t_2(h_prev_t, x_t, W_hh, W_hx, b_h):
    z = np.matmul(W_hh, h_prev_t) + np.matmul(W_hx, x_t) + b_h
    return sigmoid(z)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

folder='debug_data/'
user_operations = read_anonymize_user_operations_into_dataframe(f'{folder}anonymized_operations.csv')
endpoints = get_endpoints_from_operations(user_operations)
sequences, endpoints = get_sequence__arrays_and_vocabulary_from_operations(user_operations, sequence_width)
sequence_index_array = get_sequence_index_arrays(sequences, endpoints)
input, output = separate_input_output_from_sequences(sequence_index_array)