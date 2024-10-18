from tags import get_tags

def question_recommendations(history,LLM):
    tags, _ = get_tags(history,LLM)
    
    conversation_history = ""
    for i in range(len(history)):
        if i % 2 == 0:
            # For user messages
            user_timestamp = history[i].additional_kwargs["user_timestamp"]
            if i == 0:
                # No previous timestamp to compare for the first message
                message = f"User asked: {history[i].content}"
            else:
                assistant_timestamp = history[i-1].additional_kwargs["assistant_timestamp"]
                time_diff = user_timestamp - assistant_timestamp
                message = f"User asked after {time_diff} seconds: {history[i].content}"
        else:
            # For assistant messages
            assistant_timestamp = history[i].additional_kwargs["assistant_timestamp"]
            user_timestamp = history[i-1].additional_kwargs["user_timestamp"]
            time_diff = assistant_timestamp - user_timestamp
            message = f"Assistant replied in {time_diff} seconds: {history[i].content}"

        conversation_history += f"{i + 1}. {message}\n"
    
    recommendation_prompt_template = f"""Task: Given a conversation between the user and assistant, generate user queries, that follows the flow of the conversation. 
    The questions should be related to popular tags that the current user hasn't explored yet. If a user's current conversation has certain tags, recommend questions related to tags might co-occur with them.

    Conversation History:
    {conversation_history}

    Tags:
    {tags}

    Instructions:
    - Focus on questions that could summarize or represent the essence of the conversation and deep dive into it.
    - Return 3 questions as a simple list, separating each with a comma
    - Don't include any special characters or punctuation marks in the questions."""

    questions = LLM.complete(recommendation_prompt_template)
    questions_list = questions.text.split(", ")
    questions_list = [question.strip() for question in questions_list]
    
    return questions_list, questions