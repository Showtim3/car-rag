import pandas as pd
import glob
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")  # optional default
INDEX_NAME="car-rag"

def load_data():
  csv_files = glob.glob("vehicle-dataset-from-cardekho/*.csv")
  dfs = [pd.read_csv(f) for f in csv_files]
  df = pd.concat(dfs, ignore_index=True)
  df = df.fillna("")

  print(f"Total rows after combining: {len(df)}")
  print(df.head())

  texts = []
  for idx, row in df.iterrows():
    text = f"{row['Car_Name']} ({row.get('Year','')})\n" \
           f"Present Price: ₹{row.get('Present_Price','')}\n" \
           f"Selling Price: ₹{row.get('Selling_Price','')}\n" \
           f"Kms Driven: {row.get('Kms_Driven','')}\n" \
           f"Fuel Type: {row.get('Fuel_Type','')}\n" \
           f"Transmission: {row.get('Transmission','')}\n" \
           f"Owner: {row.get('Owner','')}"
    doc_id = f"{row['Car_Name'].replace(' ', '_')}_{row.get('Year','')}_{idx}"
    texts.append({"id": doc_id, "text": text})

  print(f"Prepared {len(texts)} car documents for ingestion.")
  return texts

pinecone = Pinecone(api_key=PINECONE_API_KEY)
def ingest_to_pinecone(texts_raw):  
  print("hello")
  print(pinecone.list_indexes())
  if INDEX_NAME not in [idx["name"] for idx in pinecone.list_indexes()]:
    pinecone.create_index(INDEX_NAME, dimension=384, spec=ServerlessSpec('aws', 'us-east-1'))

  docs = [Document(page_content=doc["text"], metadata={"id": doc["id"]}) for doc in texts_raw]

  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorstore = PineconeVectorStore(index_name="car-rag", embedding=embeddings)
  # vectorstore.add_documents(docs)
  for i, doc in enumerate(docs, start=1):
    vectorstore.add_documents([doc], ids=[doc.metadata["id"]])  # add a single document as a list
    print(f"Pushed document {i}/{len(docs)}: {doc.metadata['id']}")


def main():
  texts = load_data()
  ingest_to_pinecone(texts)

if __name__ == "__main__":
    main()