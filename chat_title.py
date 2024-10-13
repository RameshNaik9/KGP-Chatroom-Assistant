from kgpchatroom import KGPChatroomModel

def get_chat_name(user_message: str, chat_title: str) -> str:
    llm= KGPChatroomModel().get_model()
    query = f"User: {user_message}\nAssistant: {chat_title}"
    prompt = f"Summarize the following conversation between Assistant and User in 1 sentence. Make sure that the sentence is under 6 words and is a creative and catchy phrase.{query}"
    title = llm.complete(prompt)
    return title.text.strip("\n")