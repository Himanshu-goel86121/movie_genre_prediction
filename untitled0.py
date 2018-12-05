import tensorflow as tf
import pandas as pd
import keras
import json
import numpy as np
from collections import Counter
import pickle

output_padding_size = 5
embedding_size = 100
input_sequence_length = 60
output_sequence_length = 5

def process_labels(df):
    genre_list = []
    distinct_genres = []
    output_list = []
    target_sequence_size = []
    #get a list of all genres and distinct genres
    for each in df.values.tolist():
        json_str = each.replace("'", "\"")
        genre_dict = json.loads(json_str)
        genres = [genre["name"] for genre in genre_dict]
        genre_list.append(genres)
        distinct_genres.extend(genres)
    #drop duplicates
    distinct_genres = list(set(distinct_genres))
    #create a index to genre label mapping
    temp_df = pd.DataFrame()
    temp_df["distinct_genres"] = distinct_genres
    temp_df = temp_df.reset_index()
    mapping = {j: i+1 for i,j in temp_df.values.tolist()}
    #map the values
    for genre_1 in genre_list:
        temp = []
        for genre in genre_1:
            temp.append(mapping[genre])
        output_list.append(temp)
        target_sequence_size.append(len(temp))
    mapping["<GO>"] = len(mapping)
    mapping["<EOS>"] = len(mapping)
    for i in range(len(output_list)):
        if(len(output_list[i]) > 8):
            output_list[i] = output_list[i][:8]
        if(len(output_list[i]) < 8):
            output_list[i] = output_list[i] + [mapping['<EOS>']]*(8 - len(output_list[i]))
    np.save("output_mapping.npy", mapping)
    return np.array(output_list), np.array(target_sequence_size), mapping, genre_list

def process_input(df):
    tokenizer = keras.preprocessing.text.Tokenizer()
    texts = df.values.tolist()
    tokenizer.fit_on_texts(texts)
    texts = tokenizer.texts_to_sequences(texts)
    sequences = keras.preprocessing.sequence.pad_sequences(texts)
    pickle.dump(tokenizer, open("tokenizer.pkl","wb+"))
    return sequences, tokenizer.word_index

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    go_id = target_vocab_to_int['<GO>']
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat

def model_inputs():
    inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(None, None), name='targets')
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)
    #max_target_len = tf.constant(8)
    return inputs, targets, target_sequence_length, max_target_len

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size):
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])    
    outputs, state = tf.nn.dynamic_rnn(stacked_cells, 
                                       embed,
                                       dtype=tf.float32)
    return outputs, state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer 
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                               target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_summary_length)
    return outputs, max_summary_length

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer 
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], start_of_sequence_id), 
                                                      end_of_sequence_id)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)
    
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs

def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output, temp = decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output, temp)

def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output, temp = decoding_layer(dec_input,
                                               enc_states, 
                                               target_sequence_length, 
                                               max_target_sentence_length,
                                               rnn_size,
                                               num_layers,
                                               target_vocab_to_int,
                                               target_vocab_size,
                                               batch_size,
                                               keep_prob,
                                               dec_embedding_size)
    
    return train_output, infer_output, temp
        

data = pd.read_csv("movies_metadata.csv")
data = data[data["genres"] != "[]"]
data = data[pd.isnull(data["overview"]) == False]
output, target_sequence_size, output_mapping, genre_list = process_labels(data.genres)
rev_output_mapping = {j:i for i,j in output_mapping.items()}
input_seq, input_mapping = process_input(data.overview)
save_path = 'checkpoints/dev'
max_target_sentence_length = max(target_sequence_size)
train_graph = tf.Graph()
with train_graph.as_default():
    lr, keep_prob = 0.01, 0.9
    input_data, targets, target_sequence_length, max_target_sequence_length = model_inputs()
    train_logits, inference_logits, temp = seq2seq_model(input_data,
                                                   targets,
                                                   keep_prob,
                                                   1000,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(input_mapping),
                                                   len(output_mapping),
                                                   100,
                                                   100,
                                                   512,
                                                   2,
                                                   output_mapping)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits_1 = tf.identity(inference_logits.sample_id, name='predictions')
    print(training_logits)
    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10):
            average_loss = 0.0
            j=0
            while True:
                if((j+1)*1000>len(input_seq)):
                    break
                feed_dict = {input_data:input_seq[j*1000: (j+1)*1000], targets:np.array([i[:max(target_sequence_size[j*1000: (j+1)*1000].tolist())] for i in output[j*1000: (j+1)*1000].tolist()]), target_sequence_length: target_sequence_size[j*1000: (j+1)*1000]}
                print([each.shape for each in feed_dict.values()])
                #print("*"*200)
                #print(sess.run([temp], feed_dict))
                #print("*"*200)
                #1/0
                _, loss_val = sess.run([train_op, cost], feed_dict)
                print("loss at ",j," = ",loss_val)
                average_loss+=loss_val
                j+=1
                if(j%20 == 0):
                    print("Avg loss after j ", j, " is ",average_loss/20)
                    average_loss = 0.0
                print(data["overview"][j*1000: (j+1)*1000][:5])
                print(genre_list[j*1000: (j+1)*1000][:5])
                infer_logits = sess.run([inference_logits_1], feed_dict)
                print(infer_logits[0].shape)
                for each in infer_logits[0][:5]:
                    print([rev_output_mapping[i] for i in each])
                    
                    