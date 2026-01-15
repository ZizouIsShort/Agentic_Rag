from fastapi import FastAPI, Request
from ingestion.loader import load_pdf, load_hface
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

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
