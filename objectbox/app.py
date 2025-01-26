import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load the Groq And OpenAI Api Key
os.environ['OPEN_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')

st.title("Objectbox VectorstoreDB With Llama3 Demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """

)


## Vector Enbedding and Objectbox Vectorstore db

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Documents Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=768)


input_prompt=st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("ObjectBox Database is ready")

import time

if input_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()

    response=retrieval_chain.invoke({'input':input_prompt})

    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
            
# This code implements a **Streamlit application** that integrates with **LLM models** (Llama3 via ChatGroq) and **ObjectBox vector database** for document retrieval and question-answering. Here's a summary of its functionality:

# 1. **Environment Setup**:
#    - Loads API keys for Groq and OpenAI from environment variables using `dotenv`.

# 2. **LLM and Prompt Configuration**:
#    - Uses `ChatGroq` (Llama3 model) for answering questions.
#    - Defines a prompt template to instruct the model to provide answers based on specific context.

# 3. **Document Embedding and Vector Store**:
#    - Uses `OpenAIEmbeddings` to create vector embeddings for documents.
#    - Loads PDF files from a directory (`./us_census`) using `PyPDFDirectoryLoader`.
#    - Splits the documents into smaller chunks using `RecursiveCharacterTextSplitter`.
#    - Stores the processed document embeddings in an ObjectBox vector database.

# 4. **User Interaction**:
#    - A text input allows users to ask questions about the documents.
#    - A button triggers the document embedding process and initializes the ObjectBox database.

# 5. **Retrieval and Response**:
#    - Creates a retrieval chain to fetch relevant document chunks from the database.
#    - Sends the context and question to the LLM for generating answers.
#    - Displays the response along with document snippets (relevant chunks) in the UI.

# ### Key Features:
# - Efficient document retrieval using vector embeddings.
# - Seamless integration of LLM for accurate question-answering.
# - Interactive UI for embedding, querying, and viewing relevant document content.