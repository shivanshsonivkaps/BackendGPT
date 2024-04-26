from flask import Flask, render_template, jsonify, request
# from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader
import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.vectorstores import Pinecone
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

app = Flask(__name__)

def get_vectorstore():
    vector_store = PineconeStore.from_existing_index(index_name="reserch",embedding=embeddings)
    return vector_store

def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):
    time.sleep(10)
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
    )
    index_name = pinecone_index_name
    index = PineconeStore.from_existing_index(index_name, embeddings)
    return index

def similar_docs(query,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
    )
    index_name = pinecone_index_name
    index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    
    index_stat = pinecone.Index(pinecone_index_name) 
    vector_count = index_stat.describe_index_stats() 
    k = vector_count["total_vector_count"]
    
    similar_docs = index.similarity_search(query, 2)
    sources = []
    for similar_doc in similar_docs:
        metadata = similar_doc.metadata
        sources.append(metadata.get("file"))
    return sources

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()  
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ]) 
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



# PDF UPLOAD FUNCTIONS
def get_pdf_text(pdf_doc):
        text = ""
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
        )
    # create a vectorstore from the chunks
    vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=pinecone_index_name)

def create_docs(pdf_file , filename):
    # for pdf_file in pdf_files:
    docs = []
    if filename.lower().endswith(".pdf"):
        pdf_data = get_pdf_text(pdf_file)
        docs.append(Document(
            page_content=pdf_data,
            metadata={"filename": filename}
         ))
    else:
        return("invalid pdf")
          
    return docs


#API ROUTES
@app.route("/")
def home():
    return "Hello World"
@app.route("/chat", methods=["GET", "POST"])
def qa():
    if request.method == "GET":
        newJson = {
                    "status":400,
                    "message":"Please use POST req"
                }
        return jsonify(newJson),400
    if request.method == "POST":
        try:
            user_input = request.json.get("question")
            if user_input is None and user_input == "":
                newJson = {
                    "status":400,
                    "message":"Invalid question"
                }
                return jsonify(newJson),400
            vector_store = get_vectorstore()
            retriever_chain = get_context_retriever_chain(vector_store)
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            chat_history = []
            response = conversation_rag_chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            })
            source = similar_docs(user_input,PINECONE_API_KEY,"us-east-1","reserch",embeddings)
            response = response['answer'] + f"\n\nSource : {source}"

            data = {"question": user_input, "answer": response}
            
            return jsonify(data)
        except Exception as e:
            newJson = {
                    "status":400,
                    "message":"Invalid question"
                }
            return jsonify(newJson),400
@app.route('/receive_pdf', methods=["GET",'POST'])
def receive_pdf():
    try:
        #Getting files from API
         files = request.files
         file = files.get('Demo')
         file_name = file.filename
         newPdf = create_docs(file , file_name)
        #Extract embeddings and docs from the PDF file
        #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        #Push embeddings and documents to Pinecone
         pinecone_apikey = 'dfc85378-2c83-4597-8955-7e09d04dd549'
         pinecone_environment = 'us-east-1'
         pinecone_index_name = "reserch"
         push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, newPdf)
         return jsonify("Data inserted succesfully"),200
    except Exception as e:
        print(e)
        return({
            "status" : "500",
            "message" : "something wrong happened"
        }),400
    
   

if __name__ == "__app__":
    app.run(debug=True)

