from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.vectorstores import Pinecone
from pinecone import Pinecone
import os
import time

from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    vector_store = PineconeStore.from_existing_index(index_name=PINECONE_INDEX_NAME,embedding=embeddings)
    return vector_store

def push_to_pinecone(docs,pinecone_environment="us-east-1",pinecone_index_name=PINECONE_INDEX_NAME):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    pinecone = Pinecone(
        api_key=PINECONE_API_KEY,environment=pinecone_environment
        )
    # create a vectorstore from the chunks
    vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=pinecone_index_name)

def pull_from_pinecone(pinecone_environment="us-east-1"):
    time.sleep(10)
    pinecone = Pinecone(
        api_key=PINECONE_API_KEY,environment=pinecone_environment
    )
    index_name = PINECONE_INDEX_NAME
    index = PineconeStore.from_existing_index(index_name, embeddings)
    return index

def similar_docs(query,pinecone_index_name=PINECONE_INDEX_NAME,pinecone_environment="us-east-1"):
    pinecone = Pinecone(
        api_key=PINECONE_API_KEY,environment=pinecone_environment
    )
    index = pull_from_pinecone()

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

