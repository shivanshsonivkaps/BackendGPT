from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from dotenv import load_dotenv
from pinecone import Pinecone
import os
import time




#GLOBAL DECLERATIONS
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
pinecone_environment="us-east-1"



def get_vectorstore():
    """
    Functionality: Retrieves a vector store from Pinecone.
    Parameters:None.
    Description: This function fetches a vector store from Pinecone using the provided index name and embedding model.
    """
    vector_store = PineconeStore.from_existing_index(index_name=PINECONE_INDEX_NAME,embedding=embeddings)
    return vector_store


def push_to_pinecone(docs):
    """
    Pushes documents to a Pinecone index.
    Args:docs (list): A list of document objects to be pushed to Pinecone.
    """
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    pinecone = Pinecone(
        api_key=PINECONE_API_KEY,environment=pinecone_environment
        )
    vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=PINECONE_INDEX_NAME)


def pull_from_pinecone():
    """
    Pulls a Pinecone index for further processing.
    Returns:PineconeStore: A Pinecone index object.
    """
    time.sleep(10)
    index = PineconeStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    return index


def similar_docs(query):
    """
    Searches for similar documents in a Pinecone index based on a query.
    Args:query (str): The query string for which similar documents are to be searched.
    Returns:list: A list of sources of similar documents.
    """
    index = pull_from_pinecone()
    similar_docs = index.similarity_search(query, 2) 
    string_text = [similar_docs[i].page_content for i in range(len(similar_docs))]
    textual_data = string_text.pop()
    # print(textual_data)
    sources = []
    for similar_doc in similar_docs:
        metadata = similar_doc.metadata
        sources.append(metadata.get("filename")) 
    # print(sources)
    newdict = {
        "source":sources,
        "content":textual_data
    }
    return newdict


def get_context_retriever_chain(vector_store):
    """
    Constructs a retriever chain for contextual information retrieval.
    Args:vector_store (PineconeStore): The vector store from which retrieval is performed.
    Returns:retriever_chain: A retriever chain object.
    """

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
    """
    Constructs a retrieval chain for conversational response generation.
    Args:retriever_chain: The retriever chain used for contextual information retrieval.
    Returns:retrieval_chain: A retrieval chain object.
    """

    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def text_docs(query):
    """
    Searches for similar documents in a Pinecone index based on a query.
    Args:query (str): The query string for which similar documents are to be searched.
    Returns:list: A list of sources of similar documents.
    """
    index = pull_from_pinecone()
    similar_docs = index.similarity_search(query, 2) 
    string_text = [similar_docs[i].page_content for i in range(len(similar_docs))]
    textual_data = string_text.pop()
    # print(textual_data)
    sources = []
    for similar_doc in similar_docs:
        metadata = similar_doc.metadata
        sources.append(metadata.get("filename")) 
    return textual_data
