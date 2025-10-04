import pandas as pd
import glob
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

PINECONE_API_KEY=""
PINECONE_ENV=""
INDEX_NAME="car-rag"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding_function=embeddings)


def load_data():
  csv_files = glob.glob("vehicle-dataset-from-cardekho/*.csv")
  dfs = [pd.read_csv(f) for f in csv_files]
  df = pd.concat(dfs, ignore_index=True)
  df = df.fillna("")

  print(f"Total rows after combining: {len(df)}")
  print(df.head())

  texts = []
  for _, row in df.iterrows():
    text = f"{row['Car_Name']} ({row.get('Year','')})\n" \
           f"Present Price: ₹{row.get('Present_Price','')}\n" \
           f"Selling Price: ₹{row.get('Selling_Price','')}\n" \
           f"Kms Driven: {row.get('Kms_Driven','')}\n" \
           f"Fuel Type: {row.get('Fuel_Type','')}\n" \
           f"Transmission: {row.get('Transmission','')}\n" \
           f"Owner: {row.get('Owner','')}"
    texts.append(text)

  print(f"Prepared {len(texts)} car documents for ingestion.")
  return texts


def ingest_to_pinecone(texts):
  pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
  if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=384)

def ask():
  pipe = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",  # Free HF model
    temperature=0,
    max_length=512
  )
  llm = HuggingFacePipeline(pipeline=pipe)
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def ask_car(query: str):
    result = ask(query)
    answer = result['result']  # LLM response
    top_docs = [doc.page_content for doc in result['source_documents']]  # Top 3 docs
    return {"answer": answer, "top_docs": top_docs}


def main():
  texts = load_data()
  ingest_to_pinecone(texts)

if __name__ == "__main__":
    main()