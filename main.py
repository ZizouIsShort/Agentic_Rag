from fastapi import FastAPI, Request
from ingestion.loader import load_pdf, load_hface
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from google import genai
from ingestion.mkc import load_mkc, load_kms
from ingestion.embedding import load_embeddings
from ingestion.context import build_context
from ingestion.finalWalaData import mkbfinal_data
from ingestion.upsert import run_upsert
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

myren = os.getenv("myren")
pc = Pinecone(api_key=myren)
index = pc.Index("killme")
NAMESPACE = "rag-data"

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

GOOGLE_API_KEY = os.getenv("ZizouRAG")


@app.get("/documents")
def get_documents():
    return load_pdf()


@app.get("/imdonebye")
def get_imdonebye():
    return run_upsert()

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
    embedded_query = result1.embeddings[0].values

    results = index.query(
        vector=embedded_query,
        top_k=5,
        include_metadata=True,
        namespace="rag-data"
    )

    if not results.matches:
        return JSONResponse({
            "answer": "no clue based on the provided context",
            "sources": []
        })

    top_chunks = []
    for match in results.matches:
        top_chunks.append({
            "text": match.metadata["text"],
            "metadata": match.metadata,
            "score": match.score
        })

    best_score = top_chunks[0]["score"]
    print("Best Score:", best_score)

    if best_score < 0.5:
        return JSONResponse({
            "answer": "no clue based on the provided context",
            "sources": []
        })

    seen_sources = set()
    sources = []

    for chunk in top_chunks:
        meta = chunk["metadata"]
        source_type = meta.get("source_type")

        if source_type == "pdf":
            key = ("pdf", meta.get("filename"), meta.get("page_number"))
            source_entry = {
                "source_type": "pdf",
                "filename": meta.get("filename"),
                "page": meta.get("page_number")
            }

        elif source_type == "huggingface":
            key = ("huggingface", meta.get("dataset"), meta.get("row_id"))
            source_entry = {
                "source_type": "huggingface",
                "dataset": meta.get("dataset"),
                "row_id": meta.get("row_id")
            }

        else:
            continue

        if key not in seen_sources:
            seen_sources.add(key)
            sources.append(source_entry)

    final_context = build_context(top_chunks)
    print(final_context)
    final_prompt = f"""
    You are a question-answering assistant.

    RULES:
    - Answer ONLY using the provided context.
    - If the answer is not present in the context, say:
      "I don't know based on the provided context."

    CONTEXT:
    {final_context}

    QUESTION:
    {query}
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=final_prompt
    )
    print(response.text)
    return JSONResponse(
        content={
            "query": query,
            "answer": response.text,
            "similarity_score": best_score,
            "sources": sources
        }
    )


if __name__ == "__main__":
    mkbfinal_data()


