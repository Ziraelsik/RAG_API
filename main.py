# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.openapi.utils import get_openapi
import asyncio
from services.document_service import DocumentService
from services.qa_service import QAService

app = FastAPI()

document_service = DocumentService()
document_service.download_and_load_model()
qa_service = QAService(document_service=document_service)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Document QA API",
        version="1.0.0",
        description="API для загрузки документов и ответов на вопросы",
        routes=app.routes,
    )
    openapi_schema.get("components", {}).get("schemas", {}).pop("ValidationError", None)
    openapi_schema.get("components", {}).get("schemas", {}).pop("HTTPValidationError", None)
    
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            operation.get("responses", {}).pop("422", None)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = await document_service.upload_docx(file)
        return {"file_id": file_id}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/")
async def ask_question(
    request: Request,
    file_id: str = Form(...),
    question: str = Form(...)
):
    try:
        question_id = await qa_service.ask_question(file_id, question)
        return {"question_id": question_id}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/answer/{question_id}")
async def get_answer(question_id: str):
    try:
        answer_data = await qa_service.get_answer(question_id)
        return answer_data
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))