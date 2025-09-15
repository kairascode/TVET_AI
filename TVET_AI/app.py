import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- App Configuration and Title ---
st.set_page_config(page_title="TVET Kenya AI Assistant", page_icon="🇰🇪", layout="wide")
st.title("TVET Kenya AI Assistant")
st.caption("Your guide to Technical and Vocational Education and Training in Kenya.")

# --- Data Protection Act Compliance Disclaimer ---
st.warning(
    """
    **Important Notice (Data Protection Act, 2019):**
    - This is an AI assistant providing information from public TVET documents.
    - **DO NOT** enter any personal or sensitive information (e.g., names, ID numbers, contacts).
    - All conversations are processed anonymously and are not stored.
    """
)

# --- Global Variables & Constants ---
# Use a relative path for the vector database directory
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")

# --- Caching the Vector Database ---
# This ensures we don't reload and process the documents every time the app reruns.
@st.cache_resource
def load_and_process_documents():
    """
    Loads documents from the 'documents' directory, splits them into chunks,
    creates embeddings, and stores them in a Chroma vector database.
    """
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.error(f"The '{DATA_DIR}' directory is empty. Please add your PDF and .txt files.")
        return None

    # Load documents from the specified directory
    loader = DirectoryLoader(
        DATA_DIR,
        glob='**/*.pdf' if any(f.endswith('.pdf') for f in os.listdir(DATA_DIR)) else '**/*.txt',
        loader_cls=PyPDFLoader if any(f.endswith('.pdf') for f in os.listdir(DATA_DIR)) else TextLoader
    )
    documents = loader.load()

    # Split documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings (numerical representations) for the text chunks
    # OllamaEmbeddings uses the locally running Ollama model
    embeddings = OllamaEmbeddings(model="llama3")

    # Create a Chroma vector store and persist it to disk
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vectorstore

# --- Load the Vector DB ---
st.write("Initializing AI Assistant... Please wait.")
try:
    vectorstore = load_and_process_documents()
    if vectorstore is None:
        st.stop() # Stop execution if there are no documents to process
    st.success("AI Assistant is ready. Ask your question below.")
except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.stop()


# --- Define the LLM and the RAG Chain ---
# Initialize the Llama 3 model via Ollama
llm = Ollama(model="llama3")

# Create a retriever from our vector store. It will fetch relevant documents.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

# Custom Prompt Template
# This template strictly instructs the model to answer based ONLY on the provided context.
prompt_template = """
### Instruction:
You are a helpful and respectful AI assistant for the State Department for TVET in Kenya.
Your role is to answer user questions based *only* on the provided context document.
- If the answer is in the context, provide a clear and concise answer.
- If the answer cannot be found in the context, you MUST state: "I'm sorry, but I cannot find the answer to that question in the available documents. Please refer to the official TVET Kenya website."
- Do not make up information or answer from your general knowledge.
- Never ask for personal information.

### Context:
{context}

### User Question:
{question}

### Answer:
"""

QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# --- User Interface for Chat ---
query = st.text_input("**Ask a question about TVET in Kenya:**", placeholder="e.g., What are the entry requirements for a diploma course?")

if query:
    with st.spinner("Searching for an answer..."):
        try:
            # Get the response from the QA chain
            response = qa_chain.invoke({"query": query})
            
            st.subheader("Answer:")
            st.write(response["result"])

            # Optionally display the source documents for verification
            #
            #with st.expander("View Sources"):
            #st.write("The answer was generated based on the following information snippets:")
                #for doc in response["source_documents"]:
                    #st.info(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}\n\nContent: {doc.page_content}")
            
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")
