import streamlit as st
import pickle
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
@st.cache_data
def load_model_and_vectorizer():
    with open('C:\\Users\\hp\\Desktop\\EXCELR\\Resume Classification.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('C:\\Users\\hp\\Desktop\\EXCELR\\vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Define categories
categories = ['Peoplesoft resumes', 'Workday resumes', 'SQL Developer Lightning Insight', 'React Developer resumes', 'Internship resumes']

# Streamlit app
st.title('Resume Classification App')
st.write('Upload a resume to classify it into one of the following categories:')
st.write(', '.join(categories))

# Upload file
uploaded_file = st.file_uploader("Choose resume files", type=['txt', 'pdf','docx'])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.type == 'text/plain':
        resume_text = uploaded_file.read().decode('utf-8')
    elif uploaded_file.type == 'application/pdf':
        pdf_reader = PdfReader(uploaded_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = Document(uploaded_file)
        resume_text = ""
        for para in doc.paragraphs:
            resume_text += para.text + "\n"        
    else:
        st.error('Unsupported file type.')
        resume_text = ""
    
    if resume_text:
        # Display resume content
        st.subheader('Resume Content:')
        st.write(resume_text)
        
        # Vectorize the resume text
        resume_vectorized = vectorizer.transform([resume_text])
        
        # Predict category
        prediction = model.predict(resume_vectorized)
        predicted_category = prediction[0]
        
        st.subheader('Predicted Category:')
        st.write(predicted_category)
    
