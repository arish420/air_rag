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
# file_id = "1O-kyiidKyEVKgcKcOU2VpNxMdW_Ve9_2"
# output_file = "key.txt"


# https://drive.google.com/file/d/1O-kyiidKyEVKgcKcOU2VpNxMdW_Ve9_2/view?usp=sharing


# https://docs.google.com/document/d/1uBMaUsK15BvuETPtrohNJeDRMoJVRwAgIdwcmfUy7NA/edit?usp=sharing
# https://drive.google.com/file/d/1O-kyiidKyEVKgcKcOU2VpNxMdW_Ve9_2/view?usp=sharing

# https://docs.google.com/spreadsheets/d/1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g/edit?gid=0#gid=0
sheet_id = '1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g' # replace with your sheet's ID
url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df=pd.read_csv(url)
# st.write(df)


# def download_db():
#     url = f"https://drive.google.com/uc?id={file_id}"
#     gdown.download(url, output_file, quiet=False)
#     return output_file
# k=""
# with open(download_db(),'r') as f:
#     f=f.read()
#     # st.write(f)
#     k=f
# # st.write(k)
os.environ["OPENAI_API_KEY"] =  df.keys()[0]
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
#####################################################################################################################################################
# # Load all PDFs in a directory
# pdf_folder = "database"
# loader = DirectoryLoader(pdf_folder, glob="*.pdf", loader_cls=PyPDFLoader)

# # Load documents
# documents = loader.load()

# st.write(f"Loaded {len(documents)} documents from the directory")

# text_splitter = TokenTextSplitter(encoding_name='o200k_base', chunk_size=100, chunk_overlap=20)
# texts = text_splitter.split_documents(documents)
# st.write(texts)
# Assuming you have OpenAI API key set up in your environment
# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()

########################################################################### Loading the vector db ###########################################################
# Load FAISS index
# index = faiss.read_index("faiss_index.bin")
# Create embedding model
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
# # Load metadata
# with open("faiss_metadata.pkl", "rb") as f:
#     docstore_data = pickle.load(f)

# # ✅ Fix: Wrap the docstore dictionary inside InMemoryDocstore
# docstore = InMemoryDocstore(docstore_data)

# # Load index-to-docstore mapping
# with open("faiss_index_to_docstore.pkl", "rb") as f:
#     index_to_docstore_id = pickle.load(f)

# # ✅ Fix: Ensure FAISS is initialized with proper embeddings
# vector_store = FAISS(
#     index=index,
#     embedding_function=embeddings,  # ✅ Ensure embeddings are passed correctly
#     docstore=docstore,  # ✅ Wrap docstore properly
#     index_to_docstore_id=index_to_docstore_id
# )
# # Set up retriever
# retriever = vector_store.as_retriever()

# ##########################################################################setting groq api ###############################################################

# GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# from langchain_groq import ChatGroq

# llm_llama3 = ChatGroq(
#     temperature=0,
#     model="llama-3.3-70b-versatile",
#     api_key=GROQ_API_KEY
# )

# ##############################################################setting opne ai llm ##################################################################

# llm_openai = ChatOpenAI(model="gpt-4o-mini")
# ###########################################################setting RAG document formatting ##############################################################


# prompt = hub.pull("rlm/rag-prompt")
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# st.title("AI Assistant")
# # selections=st.sidebar.selectbox("☰ Menu", ["Home","AI Assistant", "Feedback"])




# custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#     You are an aviation AI assistant answering user queries based on the provided context.
    
#     Context:
#     {context}
    
#     Question:
#     {question}

#     # Instructions:
#     - Response Should be comprehensive
#     - utlilize given context as much as possible
#     - Adapt reponse according to user query
#     """
#     )






# query=""
# # tokens={}
# query=st.text_input("Write Query Here")
# if st.button("Submit") and query!="":
#     rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | custom_prompt
#     | llm_openai
#     )
#     st.subheader("Response")
#     res=rag_chain.invoke(query)
#     st.write(res.content)
#     # tokens["open_ai"]=res.response_metadata['token_usage']['total_tokens']
#     # tokens_df=pd.DataFrame(tokens.items())
#     # tokens_df.to_csv("token_usage.csv")
#     # st.write(tokens_df)















# query=st.text_input("Query")

# root_dir = "./chroma_db"
# # query = "dispatch"
# best_match = None
# highest_score = -1
# top_result = None
# if st.button("Check"):
#     for pdf_id in os.listdir(root_dir):
#         persist_dir = os.path.join(root_dir, pdf_id)
#         if os.path.isdir(persist_dir):
#             try:
#                 vectordb = Chroma(
#                     collection_name=pdf_id,
#                     embedding_function=embeddings,
#                     persist_directory=persist_dir,
#                     # client_settings=Settings(anonymized_telemetry=False)
#                 )
#                 results = vectordb.similarity_search_with_score(query, k=1)
#                 if results:
#                     doc, score = results[0]
#                     st.write(f"Matched in {pdf_id} with score {score:.4f}")
#                     if score > highest_score:
#                         best_match = pdf_id
#                         highest_score = score
#                         top_result = doc
#             except Exception as e:
#                 st.write(f"Error loading {pdf_id}: {e}")
    
#     if best_match:
#         st.write(f"\n✅ Best match: {best_match}")
#         st.write(f"🔍 Content: {top_result.page_content}")
#     else:
#         st.write("❌ No matches found.")
    

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
    
    
