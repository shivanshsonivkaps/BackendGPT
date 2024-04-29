from langchain.schema import Document
from PyPDF2 import PdfReader

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