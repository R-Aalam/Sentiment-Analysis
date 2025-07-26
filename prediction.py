# from fastapi import types
from preprocessing import *
import torch
import pandas as pd
import pickle
import dill
import sys
import types

process = Preprocessing() #calling Preprocessing class that contain clean_text and padding_text function
sys.modules['__main__'] = types.ModuleType('__main__')

# Step 2: Inject the Tokenization class into it
setattr(sys.modules['__main__'], 'Tokenization', Tokenization)

with open("vocab_obj.pkl", "rb") as f:
    token = dill.load(f)

TRAIN_MODEL = torch.jit.load("model_cpu.pt", map_location="cpu")

def batch_predict(texts,model=TRAIN_MODEL):
    text_df = pd.Series(texts)
    clean_df = text_df.apply(process.clean_text)
    encode_df = clean_df.apply(token.encode)
    padded_df = encode_df.apply(process.padding_text)

    text_tensor = torch.tensor(padded_df, dtype=torch.long)

    with torch.no_grad():
        outputs = model(text_tensor)  # raw logits
        probs = torch.softmax(outputs, dim=1)  # convert to probabilities
        preds = torch.argmax(probs, dim=1)     # get predicted class index
    return preds,probs
# batch_predict("this product is good",model=TRAIN_MODEL)

