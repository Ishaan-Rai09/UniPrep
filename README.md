# PDF to University MCQ Generator

A Streamlit web application that lets you upload PDF files and generates university-level multiple choice questions (MCQs) using Groq's AI.

## Features

- Upload multiple PDF files
- Extract text content from PDFs
- Generate university-level MCQs using Groq AI
- Customize number of questions and difficulty level
- Download generated MCQs as text files

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/pdf-mcq-generator.git
cd pdf-mcq-generator
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your Groq API key:
   - The app comes configured with a default API key in `config.py`
   - To use your own key, set it as an environment variable:
     ```
     export GROQ_API_KEY=your_groq_api_key
     ```
   - On Windows:
     ```
     set GROQ_API_KEY=your_groq_api_key
     ```
   - Or use Streamlit secrets management

## Usage

1. Run the application:
```
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload PDF files containing academic content

4. Configure the number of questions and difficulty level using the sidebar

5. Click "Generate MCQs" to process the files and create questions

6. Download the generated MCQs using the download button

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- python-dotenv
- Groq API key 