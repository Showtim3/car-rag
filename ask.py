import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from colorama import Fore, Style, init

# Load environment
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "car-rag"

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are a car expert.\n"
        "Use the following information from documents to answer the question.\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer with explanation and reasoning."
    )
)

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,  # ✅ Pass explicitly
    )
    return vectorstore

def ask(query: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Single call
    result = qa({"query": query})

    # Filter duplicates
    unique_docs = []
    seen_ids = set()
    for doc in result['source_documents']:
        if doc.metadata['id'] not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc.metadata['id'])

    answer = result['result']
    top_docs = [doc.page_content for doc in unique_docs]

    return {"answer": answer, "top_docs": top_docs}


# queries = [
#     "car with the most powerful engine",
#     "car with the most top speed",
#     "car with the highest price",
#     "cheapest car"
# ]

# for query in queries:
#     response = ask(query)
    
#     print(Fore.YELLOW + "=============")
#     print(Fore.GREEN + "Answer:\n" + Fore.CYAN + response["answer"])
    
#     print(Fore.MAGENTA + "\nTop Documents:")
#     for i, doc in enumerate(response["top_docs"], start=1):
#         print(Fore.BLUE + f"\nDocument {i}:" + Fore.WHITE)
#         print(doc)
    
#     print(Fore.YELLOW + "=============\n")
