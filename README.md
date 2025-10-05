# Car RAG App

A **Retrieval-Augmented Generation (RAG) application** for car information. Users can ask questions about cars, and the app retrieves relevant information from a database and generates accurate responses using a language model.

Using data that isnt very informative, plus dimensions are also 384 instead of the ideal 1024.

---

## Features

- **Search and Q&A**: Ask questions about car specs, pricing, features, etc.  
- **RAG Pipeline**: Combines a vector database and a language model for context-aware responses.  
- **Embeddings-based Retrieval**: Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic search.  
- **Fast and Responsive**: Handles multiple queries efficiently.

---

## Tech Stack

- Python  
- FastAPI  
- LangChain for RAG pipeline  
- Vector Database (e.g., Chroma, FAISS, Pinecone)  
- Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings  

---

## Installation

1. Clone the repository:

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt


4. Set environment variables (if required, e.g., API keys):
5. Run the app

## How it Works

1. **User query** → Sent to FastAPI endpoint.  
2. **Embedding generation** → Converts the query into a vector.  
3. **Vector search** → Retrieves relevant documents from the car dataset.  
4. **Language model** → Generates a response based on retrieved information.  
5. **Response** → Sent back to the user.

---

## Dataset

- Car specifications dataset (CSV/JSON)  
- Includes fields like `make`, `model`, `year`, `price`, `fuel_type`, `mileage`, etc.  
- Vectorized and stored in a vector database for semantic search.

---

## Contributing

1. Feel free to submit PR's.

## Contact

- **Author:** Rishabh Rawat  
- **Email:** coolrishabhrawat@gmail.com
- **GitHub:** [showtim3](https://github.com/showtim3)
