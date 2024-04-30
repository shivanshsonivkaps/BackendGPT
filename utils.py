from langchain.schema import Document
from PyPDF2 import PdfReader
import random
import string

def generate_unique_key():
    key_length = 25
    characters = string.ascii_letters + string.digits
    while True:
        key = ''.join(random.choice(characters) for _ in range(key_length))
        # Check if the key is unique
        if is_unique_key(key):
            return key
        
def is_unique_key(key):
    try:
        with open("generated_keys.txt", "r") as file:
            keys = file.read().splitlines()
            if key not in keys:
                # If the key is unique, store it for future reference
                with open("generated_keys.txt", "a") as file:
                    file.write(key + "\n")
                return True
        return False
    except FileNotFoundError:
        # If the file doesn't exist, create it and return True as the key is unique
        with open("generated_keys.txt", "w") as file:
            file.write(key + "\n")
        return True

# PDF UPLOAD
def get_pdf_text(pdf_doc):
        text = ""
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def create_docs(pdf_file , filename):
    # for pdf_file in pdf_files:
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