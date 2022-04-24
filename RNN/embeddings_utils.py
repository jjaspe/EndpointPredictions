from utils import *
from keras.models import Input, Model
from keras.layers import Dense

def get_input_output_arrays(user_operations, width):
    operations_df = get_operations_dataframe_from_list(user_operations)
    operations_by_user_and_date_df = split_operations_dataframe_per_user_and_date(operations_df)
    sequences = get_sequences_from_operations(operations_by_user_and_date_df, width)
    focus_dfs = get_focus_context_dataframes_from_sequences(sequences)
    x,y = get_arrays_from_dataframes(focus_dfs)
    return (x,y)

def get_input_output_one_hot_encodings(x_raw, y_raw, endpoints):
    unique_word_dict = create_unique_word_dict(endpoints)
    oh_x = [one_hot_encode(x, unique_word_dict) for x in x_raw]
    oh_y = [one_hot_encode(y, unique_word_dict) for y in y_raw]
    return (unique_word_dict, np.asarray(oh_x), np.asarray(oh_y) )

def get_model(input_size, embed_size:int=2):
    inp = Input(shape=(input_size,))
    x = Dense(units=embed_size, activation='linear')(inp)
    x = Dense(units=input_size, activation='softmax')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam')
    return model

def train_model(model: Model, x, y, batch_size=8, epochs=20):
    model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)
    weights = model.get_weights()[0]
    return weights

def train_embeddings(input_file, endpoints_file, embeddings_file, width = 3):
    user_operations = read_raw_user_operations(input_file)
    endpoints = read_endpoints(endpoints_file)
    raw_x, raw_y = get_input_output_arrays(user_operations, width)
    dict, oh_x, oh_y = get_input_output_one_hot_encodings(raw_x, raw_y, endpoints)
    model = get_model(oh_x.shape[1], 2)
    weights = train_model(model, oh_x, oh_y)    
    embedding_dict = {}
    for word in endpoints: 
        embedding_dict.update({
            word: weights[dict.get(word)]
            })
    return embedding_dict