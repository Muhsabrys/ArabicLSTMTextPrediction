import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the saved model
model = tf.keras.models.load_model('light_lstm_model.h5')

# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Sample input text for prediction
input_text = "في عام"

# Tokenize and pad the input text
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input = pad_sequences(input_sequence, maxlen=model.input_shape[1], padding='pre')

# Make predictions
predicted_probs = model.predict(padded_input)
predicted_index = np.argmax(predicted_probs, axis=-1)

# Convert predicted index to word
predicted_word = tokenizer.index_word.get(predicted_index[0], '')

print("Input text:", input_text)
print("Predicted word:", predicted_word)
