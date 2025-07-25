#  Text Comparison and Plagiarism Checker

## Overview
This is a powerful **Streamlit** web application designed to compare two documents and analyze their textual similarity. It supports **PDF, PPTX, DOCX, and TXT** files, providing visualizations such as **word frequency charts**, **cosine similarity heatmaps**, **word clouds**, and **scatter plots**. A **PDF report** is also generated summarizing the results.

---

##  Objectives
- Build an interactive UI for file-based text comparison.
- Extract text from various file formats: PDF, PPTX, DOCX, TXT.
- Compute text similarity and differences using NLP techniques.
- Visualize comparisons using bar charts, heatmaps, and word clouds.
- Generate downloadable PDF reports.

---

##  Features
-  Upload and compare two text-based files.
-  Word Frequency Analysis.
-  Cosine Similarity Heatmap.
-  Word Cloud Generation.
-  Scatter Plot of Word Frequencies.
-  Downloadable PDF Report.
-  Supports PDF, PPTX, DOCX, TXT formats.

---

## Technologies Used

| Category      | Technologies                         |
|---------------|--------------------------------------|
| Frontend      | Streamlit                            |
| Backend       | Python                               |
| NLP/ML        | Scikit-learn, TfidfVectorizer        |
| Visualization | Matplotlib, Seaborn, WordCloud       |
| File Parsing  | PyMuPDF, python-docx, python-pptx    |
| PDF Report    | ReportLab                            |

---

##  Project Structure

```bash
text-comparison-app/
│
├── app.py                     
├── requirements.txt           
└── README.md
```
Installation & Setup
---

Prerequisites
- Python 3.8 or higher
- pip package manager

Installation Steps
```bash
# Clone the repository
git clone https://github.com/your-username/Text-Comparison-and-Plagiarism-Detector.git

# Navigate to the project folder
cd Text-Comparison-and-Plagiarism-Detector

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
Now, open your browser and go to: http://localhost:8501

