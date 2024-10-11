from fastapi import FastAPI, HTTPException
from llama_index.core.bridge.pydantic import BaseModel
from chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from template import get_template
from model import get_index
template = get_template()
pc_index = get_index()
chat_sessions = {}
app = FastAPI()

class ChatRequest(BaseModel):
    conversation_id: str
    user_message: str

class ChatResponse(BaseModel):
    conversation_id: str
    assistant_response: str

def get_chat_engine(conversation_id: str) -> ContextChatEngine:
    if conversation_id not in chat_sessions:
        memory = ChatMemoryBuffer.from_defaults(token_limit=400000)
        chat_engine = ContextChatEngine.from_defaults(retriever=pc_index.as_retriever(), memory=memory, system_prompt=template)
        chat_sessions[conversation_id] = chat_engine
    return chat_sessions[conversation_id]

@app.post("/api/chat/{conversation_id}", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        user_message = request.user_message
        conversation_id = request.conversation_id
        # chat_profile = request.chat_profile

        # Get the chat engine associated with this conversation
        chat_engine = get_chat_engine(conversation_id)

        # Call the chat method to get a direct response
        response = chat_engine.chat(user_message)

        # Return the response in JSON format
        return ChatResponse(conversation_id=conversation_id, assistant_response=str(response))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_reset/{conversation_id}")
def reset_chat(conversation_id: str):
    try:
        if conversation_id in chat_sessions:
            chat_sessions[conversation_id].reset()  # Reset memory for that conversation
            return {"message": f"Chat history for conversation {conversation_id} has been reset successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Conversation ID {conversation_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# To run the app:
# Use: uvicorn original_code:app --reload
