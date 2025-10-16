import os
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization
import pickle

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "AIDetect.h5")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

# 載入詞表
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

vectorized_layer = TextVectorization(
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=50,
    ragged=True
)
vectorized_layer.set_vocabulary(vocab)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/aiDetect")
def predict(data: TextInput):
    seq = vectorized_layer(tf.constant([data.text]))
    padded = pad_sequences(seq.to_list(), maxlen=50, padding='pre', truncating='pre')
    preds = model.predict(padded)
    score = float(preds[0][0])
    return {"prediction_score": score}