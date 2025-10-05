import uvicorn
from ask import ask
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Car RAG API", version="1.0")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    top_docs: List[str]

templates = Jinja2Templates(directory="templates")


@app.post("/ask", response_model=QueryResponse)
def ask_car_question(request: QueryRequest):
    result = ask(request.question)
    return QueryResponse(answer=result["answer"], top_docs=result["top_docs"])

@app.get("/ask")
def ask_car_question(question: str= ""):
    print("question", question)
    result = ask(question)
    return QueryResponse(answer=result["answer"], top_docs=result["top_docs"])

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/hello")
def hello():
    return "Hello"

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)