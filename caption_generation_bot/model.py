#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:03:16 2020

@author: evgenii
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import os
from tensorflow import keras
from keras_utils import reset_tf_session
import utils
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 299
CHECKPOINT_ROOT = ""
def get_checkpoint_path(epoch=None):
    if epoch is None:
        return os.path.abspath(CHECKPOINT_ROOT + "captions_weights/weights")
    else:
        return os.path.abspath(CHECKPOINT_ROOT + "captions_weights/weights_{}".format(epoch))
    
# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    # keras.backend.set_learning_phase(False) #keras backend
    model = keras.applications.InceptionV3(include_top=False)
    model.training = False
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model

import pickle
with open('vocabulary/vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

with open('vocabulary/vocab_inverse.pickle', 'rb') as handle:
    vocab_inverse = pickle.load(handle)


PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

IMG_EMBED_SIZE = 2048
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = vocab[PAD]

s = reset_tf_session()
tf.set_random_seed(42)
class decoder:
    # [batch_size, IMG_EMBED_SIZE]
    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
    # [batch_size, time steps]
    sentences = tf.placeholder('int32', [None, None])
    
    # image embedding -> bottleneck
    img_embed_to_bottleneck = keras.layers.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')
    # image embedding bottleneck -> lstm initial state
    img_embed_bottleneck_to_h0 = keras.layers.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
    # word -> embedding
    word_embed = keras.layers.Embedding(len(vocab), WORD_EMBED_SIZE)
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    
    # lstm output -> logits bottleneck
    token_logits_bottleneck = keras.layers.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
    # logits bottleneck -> logits for next token prediction
    token_logits = keras.layers.Dense(len(vocab),
                           input_shape=(None, LOGIT_BOTTLENECK))

    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))
    word_embeds = word_embed(sentences[:, :-1]) 

    hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

    flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS])

    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states)) 

    flat_ground_truth = tf.reshape(sentences[:, 1:], [-1,]) 

    flat_loss_mask = tf.not_equal(flat_ground_truth, pad_idx)

    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth, 
        logits=flat_token_logits
    )

    loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask)) 

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(decoder.loss)


saver = tf.train.Saver()

s.run(tf.global_variables_initializer())

class final_model:
    # CNN encoder
    encoder, preprocess_for_model = get_cnn_encoder()
    saver.restore(s, get_checkpoint_path())  # keras applications corrupt our graph, so we restore trained weights
    
    # containers for current lstm state
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    # input images
    input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

    # get image embeddings
    img_embeds = encoder(input_images)

    # initialize lstm state conditioned on image
    init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)
    
    # current word index
    current_word = tf.placeholder('int32', [1], name='current_input')

    # embedding for current word
    word_embed = decoder.word_embed(current_word)

    # apply lstm cell, get new lstm states
    new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

    # compute logits for next token
    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))
    # compute probabilities for next token
    new_probs = tf.nn.softmax(new_logits)

    # `one_step` outputs probabilities of next token and updates lstm hidden state
    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)

# this is an actual prediction loop
def generate_caption(image, t=1, sample=False, max_len=20):
    # condition lstm on the image
    s.run(final_model.init_lstm, 
          {final_model.input_images: [image]})
    
    # current caption
    # start with only START token
    caption = [vocab[START]]
    
    for _ in range(max_len):
        next_word_probs = s.run(final_model.one_step, 
                                {final_model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()
        
        # apply temperature
        next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break
       
    return list(map(vocab_inverse.get, caption))

# look at validation prediction example
def apply_model_to_image_raw_bytes(raw):
    img = utils.decode_image_from_buf(raw)
    fig = plt.figure(figsize=(7, 7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    print(' '.join(generate_caption(img)[1:-1]))
    answ = ' '.join(generate_caption(img)[1:-1])
    plt.show()
    return answ

