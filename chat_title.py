from kgpchatroom import KGPChatroomModel

def get_chat_name(user_message: str, title_response: str) -> str:
    llm= KGPChatroomModel().get_model()
    query = f"User: {user_message}\nAssistant: {title_response}"
    prompt = f"Based on the following text, create a concise and engaging chat title that captures the core theme or main idea. Keep it under 5 words. Ensure it is descriptive yet intriguing, encouraging further exploration of the topic.{query}"
    title = llm.complete(prompt)
    return title.text.strip("\n")