import joblib
from preprocess import clean_text

model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

def predict_message(text):
    cleaned = clean_text(text)
    x = vectorizer.transform([cleaned])
    pred = model.predict(x)[0]
    return pred

if __name__ == "__main__":
    while True:
        msg = input("Enter a message (or type 'quit'): ")
        if msg.lower() == "quit":
            break
        print("Prediction:", predict_message(msg))
