import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
import string
import PyPDF2
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Fix for punkt_tab resource issue
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define text cleaning function
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Load and prepare dataset
def load_and_train_model():
    dataset_path = 'resume_dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found. Please place it in the project directory.")

    # Load dataset
    resumeDataSet = pd.read_csv(dataset_path, encoding='utf-8')

    # Clean resumes
    resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))

    # Encode categories
    global le
    le = LabelEncoder()
    resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

    # Vectorize text
    requiredText = resumeDataSet['cleaned_resume'].values
    requiredTarget = resumeDataSet['Category'].values

    global word_vectorizer
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500
    )
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)

    # Train model
    global clf
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)

    # Evaluate model
    prediction = clf.predict(X_test)
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

    return clf, word_vectorizer, le

# Function to predict category for a resume
def predict_resume_category(file_path):
    text = extract_text_from_pdf(file_path)
    if text:
        cleaned_text = cleanResume(text)
        text_features = word_vectorizer.transform([cleaned_text])
        prediction = clf.predict(text_features)
        predicted_category = le.inverse_transform(prediction)[0]
        return predicted_category
    else:
        return None

# Flask routes
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_category = predict_resume_category(file_path)
        if predicted_category:
            return render_template('result.html', category=predicted_category, filename=filename)
        else:
            return render_template('result.html', error="Could not process the resume. Please ensure the PDF contains extractable text.")
    else:
        return render_template('result.html', error="Invalid file format. Please upload a PDF file.")

# Main execution
if __name__ == '__main__':
    # Train model and initialize globals
    clf, word_vectorizer, le = load_and_train_model()

    # Option for command-line PDF input
    print("\nOptionally, enter the path to a PDF resume for prediction (or leave blank to use web interface):")
    file_path = input().strip()
    if file_path and os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
        predicted_category = predict_resume_category(file_path)
        if predicted_category:
            print(f"Predicted Job Category: {predicted_category}")
        else:
            print("Could not process the resume. Please ensure the PDF contains extractable text.")

    # Start Flask app
    app.run(debug=True, use_reloader=False)