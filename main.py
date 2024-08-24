import os
import streamlit as st
import pickle
import time
import langchain

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_community.vectorstores.faiss import FAISS


# Importing for logging purpose
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('app.log'),
                        logging.StreamHandler()
                        ]
                    )

# Get logger
logger = logging.getLogger(__name__)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.title("News Research Tools 📈")
st.sidebar.title("New Article URLs")

urls = []
for idx in range(3):
    url = st.sidebar.text_input(f"URL {idx + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_index.pkl"


main_placeholder = st.empty()
llm = OpenAI(temperature=0.6, max_tokens=500)

if process_url_clicked:
    # Loading datas
    logger.info('Started data loading using Unstructured URL Loader')
    main_placeholder.text("Started data loading.....")
    url_loader = UnstructuredURLLoader(urls= urls)
    data = url_loader.load()

    # Splitting data to create chunks
    logger.info('Splitting data to create chunks using RecursiveCharacterTextSplitter...')
    main_placeholder.text("Started text splitter....")
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators= ['\n\n', '\n', '.', ','],
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    documents = recursive_text_splitter.split_documents(data)

    logger.info('Create embeddings for these chunks and save them to FAISS index...')
    # Create embeddings for these chunks and save them to FAISS index
    main_placeholder.text("Started building embeedding vector")
    openai_embeddings = OpenAIEmbeddings()
    vector_index_openai = FAISS.from_documents(documents, openai_embeddings)

    logger.info('Saved serialiezed vector index into a pickle file...')
    vector_index_serialized = vector_index_openai.serialize_to_bytes()
    with open(file_path, "wb+") as dump_file:
        pickle.dump(vector_index_serialized, dump_file)

    st.session_state.urls_processed = True

    # After URL Processing is completed, get the input query from the user.

if st.session_state.get("urls_processed", False):
    query = st.text_input("Type your question here:")
    submit_query = st.button("Submit Question")
    logger.info(f"Query: {query}")

    if submit_query and query:
        logger.info('Loading saved serialiezed vector index from pickle file...')
        if os.path.exists(file_path):
            with open(file_path, "rb+") as load_file:
                vector_index_serialized = pickle.load(load_file)
    
                logger.info('Deserializing the loaded serialized vectors...')
                vector_index_openai = FAISS.deserialize_from_bytes(
                    serialized= vector_index_serialized,
                    embeddings = OpenAIEmbeddings(),
                    allow_dangerous_deserialization= True
                )
    
                # Retrieve similar embeddings for a given question and call LLM to retrieve final answer
                logger.info('Retrieving similar embeddings for the given questions')
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm = llm, 
                    retriever = vector_index_openai.as_retriever()
                )
    
                logger.info('Preparing answers to the given questions...')
                result = chain.invoke({"question": query}, return_only_outputs=True)
                st.header("Answer")
                st.write(result["answer"])
        
                # Display sources
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)
    else:
        st.warning("Please enter a question")