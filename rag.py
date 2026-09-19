from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

loader = PyPDFLoader(file_path="Documents/2608.08801v1.pdf", mode="page")
docs = loader.load()
docs[0].metadata["page_label"]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=55, length_function=len
)
texts = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="embeddinggemma:latest")

vector_store = Chroma.from_documents(
    documents=texts,
    collection_name="rag_collection",
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",
)

query = input("What do you want to know about?")

results = vector_store.similarity_search(query, k=3)

prompt_template = ChatPromptTemplate.from_template(
    """ You are a helpful assistant answering questions about a documents.
    Answer the question strictly based ONLY on the following provided context. If the information is insufficient to answer the question, state 'I don't know based on this video.'
    Context: {context} 
    Question: {question} 
    Answer: """
)

llm = ChatOllama(model="llama3.2:1b", reasoning=None, validate_model_on_init=True)

chain = prompt_template|llm

rag_results = chain.invoke({"context":results, "question":query})

rag_results.content