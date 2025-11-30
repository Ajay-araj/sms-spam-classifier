# SMS Spam Classifier

**Project:** SMS Spam Classifier — TF-IDF + Multinomial Naive Bayes  
**Author:** AJAYA RAJ A N  
**Live demo:** (Add your Streamlit Cloud / HuggingFace link here after deployment)

## Summary
A simple SMS spam classifier that detects whether an SMS is **spam** or **ham** (not spam). The project includes:
- Data preprocessing (cleaning, stopwords, stemming)
- TF-IDF feature extraction
- Multinomial Naive Bayes classifier
- Streamlit web app for interactive testing and dataset preview

## Folder structure
\\\
sms-spam-classifier/
├── app/                       # Streamlit app
│   └── streamlit_app.py
├── data/
│   └── spam.csv               # dataset used for training
├── models/
│   ├── model.joblib           # trained model (optional)
│   └── vectorizer.joblib
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── scripts/                   # helper scripts (optional)
├── requirements.txt
└── README.md
\\\

## Setup (local)
1. Create & activate virtual environment:
\\\
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
\\\

2. Install dependencies:
\\\
pip install -r requirements.txt
\\\

3. Train the model (creates \models/model.joblib\ and \models/vectorizer.joblib\):
\\\\
python src/train.py
\\\

4. Run the web app:
\\\
streamlit run app/streamlit_app.py
\\\

## Usage
- Open the Streamlit URL printed in the terminal (usually \http://localhost:8501\).
- Type a message and press **Predict** to get label + confidence.

## Notes on deployment
- Ensure `requirements.txt` is present
- Make sure the app entry file is: `app/streamlit_app.py`
- If models are ignored in `.gitignore`, upload `models/` manually to deployment platform

---

## Author
AJAYA RAJ A N  
LinkedIn: https://www.linkedin.com/in/ajayaraj98  
GitHub: https://github.com/Ajay-araj

