from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
from Retriever import *
from ReaderLLM import *

class ChatBot:
    def __init__(self):
        self.app = FastAPI()
        self.ragModel = Retriever(file_path='./data/bridge.csv')
        self.reader_llm = ReaderLLM()

        @self.app.post("/chat")
        async def chat(request: self.ChatRequest) -> Dict[str, str]:
            try:
                prompt = request.prompt
                response_text = self.reader_llm.query_rag(self.ragModel, prompt)

                return {"response": response_text}

            except Exception as e:
                return {"error": str(e)}

    class ChatRequest(BaseModel):
        prompt: str

    def chatbot_response(self, user_input: str) -> str:
        response = f"You said: {user_input}"
        return response

app = ChatBot().app