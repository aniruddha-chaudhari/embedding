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
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        db.add_to_database(temp_path)
        
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
