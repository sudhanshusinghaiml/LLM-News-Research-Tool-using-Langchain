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
    main_placeholder.text("Started data loading.....")
    url_loader = UnstructuredURLLoader(urls= urls)
    data = url_loader.load()

    # Splitting data to create chunks
    main_placeholder.text("Started text splitter....")
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators= ['\n\n', '\n', '.', ','],
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    documents = recursive_text_splitter.split_documents(data)

    # Create embeddings for these chunks and save them to FAISS index
    main_placeholder.text("Started building embeedding vector")
    openai_embeddings = OpenAIEmbeddings()
    vector_index_openai = FAISS.from_documents(documents, openai_embeddings)

    vector_index_serialized = vector_index_openai.serialize_to_bytes()
    with open(file_path, "wb+") as dump_file:
        pickle.dump(vector_index_serialized, dump_file)


    query = main_placeholder.text_input("Question: ")

    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb+") as load_file:
                vector_index_serialized = pickle.load(load_file)

                vector_index_openai = FAISS.deserialize_from_bytes(
                    serialized= vector_index_serialized,
                    embeddings= openai_embeddings,
                    allow_dangerous_deserialization= True
                )
    

                # Retrieve similar embeddings for a given question and call LLM to retrieve final answer
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm = llm, 
                    retriever = vector_index_openai.as_retriever()
                )

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
