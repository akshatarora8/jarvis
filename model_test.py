import json
import pickle
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents JSON
with open("intents.json") as file:
    data = json.load(file)

# Load trained model
model = load_model("chat_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Chat loop
while True:
    input_text = input("Enter your command-> ")
    
    # Convert input to tokenized sequence and pad it
    padded_sequences = pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=20, truncating='post')
    
    # Predict intent
    result = model.predict(padded_sequences)
    tag = label_encoder.inverse_transform([np.argmax(result)])[0]

    # Find and print the response
    for i in data['intents']:
        if i['tag'] == tag:
            print(random.choice(i['responses']))
            break
