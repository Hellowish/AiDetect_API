import os
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization
import pickle

MODEL_PATH = r"C:\Users\p0931\OneDrive\文件\OneDrive\桌面\kaggleAPI\AIDetect.h5"
VOCAB_PATH = r"C:\Users\p0931\OneDrive\文件\OneDrive\桌面\kaggleAPI\vocab.pkl"  # 你先用 notebook 生成這個詞表檔案

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