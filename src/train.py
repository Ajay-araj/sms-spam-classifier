import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from preprocess import clean_text

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/spam.csv", encoding="latin-1")

df.columns = ['label', 'text']
df['text_clean'] = df['text'].apply(clean_text)

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tf, y_train)

y_pred = model.predict(X_test_tf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
print("Model saved successfully!")
