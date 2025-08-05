# from chromadb.config import Settings

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores import Chroma
import streamlit as st
import pandas as pd
import sqlite3
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import getpass
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import sentence_transformers
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from datetime import datetime
import gdown
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore  # ✅ Correct
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Create embedding model
st.title("AIR Assistant")

# Load environment variables
load_dotenv()
###############################################################setting openai ai api##################################################################
sheet_id = '1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g' # replace with your sheet's ID
url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df=pd.read_csv(url)
os.environ["OPENAI_API_KEY"] =  df.keys()[0]
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

##########################################################################################################################################

def custom_prompt(context, question):
    return f"""
    You are an aviation AI assistant answering user queries based on the provided context.
    
    Context:
    {context}
    
    Question:
    {question}

    # Instructions:
    - Response Should be comprehensive.
    - utlilize given context as much as possible.
    - anser each aspect of query.
    - Adapt reponse according to user query.
    """


conn = sqlite3.connect("embeddings.db")
cursor = conn.cursor()

cursor.execute("SELECT file_name, text, embedding FROM embeddings")
rows = cursor.fetchall()

# Reconstruct DataFrame
data = []
for file_name, text, emb_blob in rows:
    embedding = pickle.loads(emb_blob)
    data.append({"file_name": file_name, "text": text, "embedding": embedding})

df_loaded = pd.DataFrame(data)
conn.close()




query= st. text_input("Write Concern Here")


if st.button("Ask") and query!="":
    query_embedding = np.array(embed_model.embed_query(query)).reshape(1, -1)
    
    # Create embedding matrix
    embedding_matrix = np.vstack(df_loaded['embedding'].values)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]
    df_loaded["similarity"] = similarities
    
    # Get top 5 similar files
    top5 = df_loaded.sort_values(by="similarity", ascending=False)
    # print(top5[["file_name", "similarity"]])


    # Initialize GPT-4o-mini model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    # Use `.invoke()` with a message
    response = llm.invoke(custom_prompt("\n".join(top5['text'][:5]),query))
    
    st.write(response.content)
    
    
