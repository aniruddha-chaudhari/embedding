from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from vectorEmbed import VectorDatabase
from typing import Optional

class TextRequest(BaseModel):
    text: str
    source_id: str

class QueryRequest(BaseModel):
    query: str
    source: Optional[str] = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = VectorDatabase()

@app.post("/add-text")
async def add_text(request: TextRequest):
    try:
        db.add_content_to_database(request.text, request.source_id)
        return {"message": "Text processed successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "details": {
                    "source_id": request.source_id,
                }
            }
        )

@app.post("/query")
async def query(request: QueryRequest):
    try:
        results = db.query_database(request.query, request.source)
        return {"results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "query": request.query,
                "source": request.source
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
