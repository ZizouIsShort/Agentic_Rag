from fastapi import FastAPI, Request
from ingestion.loader import load_pdf, load_hface
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from google import genai
from ingestion.mkc import load_mkc, load_kms
import os
from ingestion.embedding import load_embeddings

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

GOOGLE_API_KEY = os.getenv("ZizouRAG")


@app.get("/documents")
def get_documents():
    return load_pdf()


@app.get("/bed")
def get_bed():
    return load_embeddings()


@app.get("/embed")
def get_embed():
    dataset = load_kms()
    tex=""
    for chunk in dataset:
        tex = chunk["text"]
        if tex is None:
            continue

    client = genai.Client(api_key=GOOGLE_API_KEY)
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=tex
    )
    print(dataset)
    print(result)
    return result


@app.get("/test")
def get_hface():
    return load_hface()


@app.get("/zizou")
def get_zizou():
    return load_mkc()


@app.get("/pizou")
def get_zizou():
    return load_kms()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/myre")
def myre(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    query = body.get("query")
    print("Received Request : ", query)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    result1 = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=query
    )
    print(response.text)
    print("PAUSE")
    print(result1)

    return JSONResponse(
        content={
            "status": "received",
            "data": body
        })
