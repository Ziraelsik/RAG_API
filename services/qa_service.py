# services/qa_service.py
import uuid
import asyncio
from typing import Dict
from fastapi import HTTPException
from openai import OpenAI as OpenAIClient
from config.settings import QWEN_MODEL, OPENROUTER_API_KEY

class QAService:
    def __init__(self, document_service):
        self.questions_store: Dict[str, Dict] = {}
        self.document_service = document_service
        self.llm_client = OpenAIClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

    async def ask_question(self, file_id: str, question: str) -> str:
        if not file_id or not question:
            raise HTTPException(status_code=400, detail="file_id и question обязательны")
            
        _ = self.document_service.get_vectorstore(file_id)
        question_id = str(uuid.uuid4())
        self.questions_store[question_id] = {
            "file_id": file_id,
            "question": question,
            "answer": None,
            "status": "processing"
        }
        asyncio.create_task(self.process_question(question_id))
        return question_id

    async def get_answer(self, question_id: str) -> Dict[str, str]:
        if not question_id:
            raise HTTPException(status_code=400, detail="question_id обязателен")
            
        if question_id not in self.questions_store:
            raise HTTPException(status_code=404, detail="Вопрос не найден")
            
        record = self.questions_store[question_id]
        if record["status"] == "processing":
            return {"answer": "", "status": "processing"}
        return {"answer": record["answer"], "status": "ready"}

    async def process_question(self, question_id: str):
        try:
            record = self.questions_store[question_id]
            file_id = record["file_id"]
            question = record["question"]
            vector_store = self.document_service.get_vectorstore(file_id)

            retrieved_docs = vector_store.similarity_search(question, k=3)
            contexts = [doc.page_content for doc in retrieved_docs]

            prompt = f"Оцени релевантность следующих текстов по шкале от 0 до 1 для вопроса: '{question}'. Для каждого текста дай оценку и краткое объяснение.\n\n"
            for i, ctx in enumerate(contexts):
                prompt += f"Текст {i+1}:\n{ctx}\n\n"

            ranking_completion = self.llm_client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            best_context = contexts[0] if contexts else ""

            answer_prompt = f"Используя следующий контекст, дай краткий и точный ответ на вопрос: {question}\n\nКонтекст:\n{best_context}"

            answer_completion = self.llm_client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": answer_prompt}]
            )
            answer = answer_completion.choices[0].message.content

            record["answer"] = answer
            record["status"] = "ready"
        except Exception as e:
            record["answer"] = f"Ошибка при обработке вопроса: {str(e)}"
            record["status"] = "error"