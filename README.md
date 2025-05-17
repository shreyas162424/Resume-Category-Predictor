# 📄 Resume Category Predictor

A machine learning web app that classifies uploaded PDF resumes into job categories using natural language processing and a trained K-Nearest Neighbors model. Built with Flask, Scikit-learn, and NLTK.

## 🚀 Features

- Upload a resume in `.pdf` format
- Extracts and cleans resume text automatically
- Predicts job category using a trained K-Nearest Neighbors classifier
- Clean and simple web interface

## 📂 Project Structure

```
├── resume.py                # Main Flask app with ML pipeline
├── Resume_Screening.ipynb   # Jupyter notebook for experimentation
├── resume_dataset.csv       # Labeled dataset of resumes
├── templates/
│   ├── upload.html          # HTML form for uploading resume
│   └── result.html          # Displays prediction result
├── uploads/                 # Folder to store uploaded resumes
```

## 🧰 Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas & NumPy
- NLTK (Natural Language Toolkit)
- TfidfVectorizer
- PyPDF2 (PDF text extraction)
- HTML5/CSS3

## 📊 Dataset

The project uses `resume_dataset.csv` which contains:
- **Resume**: The raw resume text
- **Category**: The labeled job category

## ⚙️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/resume-category-predictor.git
cd resume-category-predictor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python resume.py
```

4. **Open your browser**

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) and upload a resume to get started.

## 🧠 How It Works

- PDF resumes are uploaded via the web form.
- Text is extracted using `PyPDF2`.
- Text is cleaned and tokenized using `nltk`.
- Features are extracted using `TfidfVectorizer`.
- Classification is done using `KNeighborsClassifier` in a One-vs-Rest scheme.
- Result is displayed with predicted job category.

## 📌 Notes

- Only PDF files are supported.
- Max file size allowed is 16MB.
- Ensure that the PDF has extractable text (not scanned images).

## 📷 Screenshots

**Upload Page:**

![Upload Screenshot](screenshots/upload.png)

**Result Page:**

![Result Screenshot](screenshots/result.png)

## 👤 Author

**Made by Shreyas Sangalad**
