from fastapi import FastAPI, Request
from ingestion.loader import load_pdf, load_hface
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from google import genai
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

GOOGLE_API_KEY = os.getenv("ZizouRAG")


@app.get("/documents")
def get_documents():
    return load_pdf()


@app.get("/test")
def get_hface():
    return load_hface()


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
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=query
    )
    print(response.text)
    return JSONResponse(
        content={
            "status": "received",
            "data": body
        })
