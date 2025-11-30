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
\\\ash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
\\\

2. Install dependencies:
\\\ash
pip install -r requirements.txt
\\\

3. Train the model (creates \models/model.joblib\ and \models/vectorizer.joblib\):
\\\ash
python src/train.py
\\\

4. Run the web app:
\\\ash
streamlit run app/streamlit_app.py
\\\

## Usage
- Open the Streamlit URL printed in the terminal (usually \http://localhost:8501\).
- Type a message and press **Predict** to get label + confidence.

## Notes on deployment
- If you keep \models/\ in \.gitignore\, you must either push trained models manually or retrain within the deployment environment.
- For Streamlit Cloud / HuggingFace Spaces: ensure \equirements.txt\ exists and \pp/streamlit_app.py\ is set as the main file.

## Author
AJAYA RAJ A N  
LinkedIn: https://www.linkedin.com/in/ajayaraj98  
GitHub: https://github.com/Ajay-araj

