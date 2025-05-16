import os
import uuid
from typing import Dict
from fastapi import HTTPException, UploadFile
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import docx2txt
from config.settings import MODEL_REPO

class DocumentService:
    def __init__(self, model_repo: str = MODEL_REPO):
        self.documents_store: Dict[str, FAISS] = {}
        self.model_repo = model_repo
        self.local_model_path = None
        self.embeddings = None

    def download_and_load_model(self):
        os.makedirs("models", exist_ok=True)
        local_dir = f"models/{self.model_repo.replace('/', '_')}"
        if os.path.exists(local_dir) and os.listdir(local_dir):
            self.local_model_path = local_dir
        else:
            self.local_model_path = snapshot_download(
                repo_id=self.model_repo,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.local_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    async def upload_docx(self, file: UploadFile) -> str:
        if self.embeddings is None:
            raise HTTPException(status_code=500, detail="Embedding model не загружена")
        
        if not file.filename.lower().endswith(".docx"):
            raise HTTPException(status_code=400, detail="Только .docx файлы поддерживаются")
        
        try:
            text = docx2txt.process(file.file)
            chunks = [chunk for chunk in text.split('\n\n') if chunk.strip()]
            if not chunks:
                raise HTTPException(status_code=400, detail="Документ не содержит текста")
                
            vector_store = FAISS.from_texts(chunks, self.embeddings)
            file_id = str(uuid.uuid4())
            self.documents_store[file_id] = vector_store
            return file_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

    def get_vectorstore(self, file_id: str) -> FAISS:
        if file_id not in self.documents_store:
            raise HTTPException(status_code=404, detail="Файл не найден")
        return self.documents_store[file_id]