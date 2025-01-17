from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from vectorEmbed import VectorDatabase

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = VectorDatabase()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read file content directly
        file_content = await file.read()
        
        # Process the content
        content = db.read_file_content(file_content, file.filename)
        db.add_content_to_database(content, file.filename)
        
        return {"message": "File processed successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "details": {
                    "file_name": file.filename,
                    "content_type": file.content_type
                }
            }
        )

@app.post("/query")
async def query(query: str = Form(...)):
    try:
        results = db.query_database(query)
        return {"results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "query": query
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
