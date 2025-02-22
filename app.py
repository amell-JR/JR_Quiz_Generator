import streamlit as st
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
from transformers import pipeline

# Load Mistral-7B-Instruct model for text generation
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

llm = load_model()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text")
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract text from images using OCR
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# Generate Quiz Questions
def generate_quiz(text):
    prompt = f"Generate 5 multiple-choice questions based on the following text:\n{text[:1000]}"  # Limit text size
    response = llm(prompt, max_length=512, num_return_sequences=1)
    return response[0]['generated_text']

# Streamlit UI
st.title("📚 AI Quiz Generator")
st.write("Upload a PDF, DOCX, or Image, and let AI generate a quiz from the content!")

uploaded_file = st.file_uploader("Upload a document or image", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    st.subheader("Extracted Text")
    
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        extracted_text = extract_text_from_image(uploaded_file)
    
    st.text_area("Extracted Text", extracted_text, height=200)
    
    if st.button("Generate Quiz"):
        st.subheader("Quiz Questions")
        quiz = generate_quiz(extracted_text)
        st.write(quiz)

