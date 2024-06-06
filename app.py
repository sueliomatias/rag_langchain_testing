from openai import OpenAI
import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

BASE_URL_LOCAL = "http://10.10.8.39:11434"
MODEL = "mistral"

# carregar página web
def loadWebPage(url):
    loader = WebBaseLoader(
        web_paths=(url),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("btn-second")
            )
        ),
    )
    docs = loader.load()
    return docs

# carregar PDF
def loadPDF():
    loader = PyPDFDirectoryLoader("./pdf")
    docs = loader.load()
    return docs

# dividir documentos
def splitDocs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

# carregar documentos e dividir em partes
docs = loadPDF()
splits = splitDocs(docs)
print(f"Loaded {len(splits)} documents")

# carregar embeddings
embeddings = OllamaEmbeddings(model=MODEL, base_url=BASE_URL_LOCAL)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# função para instanciar o modelo de linguagem e responder a pergunta
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    clientOpen = OpenAI(api_key='sk-...', base_url=BASE_URL_LOCAL + "/v1")
    completion = clientOpen.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": formatted_prompt},
        ],
    )
    return completion.choices[0].message.content

# função para combinar documentos
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# função para recuperar documentos
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# função para fazer a pergunta
result = rag_chain("Do que se trata o documento?")
print(result)

