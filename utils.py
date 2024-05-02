from langchain.schema import Document
from PyPDF2 import PdfReader
import uuid

def generate_random_string():
    # Generate a UUID and remove dashes, then take the first 10 characters
    return str(uuid.uuid4().hex)[:10]

def get_pdf_text(pdf_doc):
        text = ""
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def create_docs(pdf_file , filename):
    
    docs = []
    if filename.lower().endswith(".pdf"):
        pdf_data = get_pdf_text(pdf_file)
        if not pdf_data:
            raise ValueError("No text found in PDF file")
       
        docs.append(Document(
            page_content=pdf_data,
            metadata={"filename": filename}
         ))
    else:
        return("invalid pdf")

    return docs 

def store_text_to_file(text):
    with open("dataset.txt", 'w') as file:
        file.write(text)



