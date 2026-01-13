from fastapi import FastAPI
from ingestion.loader import load_pdf

app = FastAPI()

@app.get("/documents")
def get_documents():
    return load_pdf()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
