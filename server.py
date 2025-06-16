from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("wordnet")

# Load model and vectorizer
with open("support_vector.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# FastAPI app
app = FastAPI()

class InputText(BaseModel):
    text: str

def clean_text(text):
    text = text.strip().lower()
    text = ''.join(c for c in text if c not in string.punctuation and c.isascii() and not c.isdigit())
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@app.get("/")
def read_root():
    return {"message": "ML model deployment successful"}

@app.post("/predict")
def predict(input: InputText):
    cleaned = clean_text(input.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "Positive Review" if prediction == 1 else "Negative Review"
    return {"prediction": label}
