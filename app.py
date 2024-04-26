from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import os
import time
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.vectorstores import Pinecone
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
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

import re
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

@app.route("/")
def home():
    return "Hello World"

@app.route("/chat", methods=["GET", "POST"])
def qa():
    if request.method == "POST":
        try:
            user_input = request.json.get("question")
            print(user_input)
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
                    "message":e
                }
            return jsonify(newJson),400

app.run(debug=True, port=5001)
