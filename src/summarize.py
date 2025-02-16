import ollama
from pydantic import BaseModel as pydantic_BaseModel

# FIXME: use mlx from hugginface instead of ollama

MODELS = {
    'fast': 'deepseek-r1:7b',      # 7B params, ~4GB RAM
    'balanced': 'deepseek-r1:14b', # 14B params, ~10GB RAM
    'quality': 'deepseek-v3:70b'   # Largest available locally
}


class ConversationSummary(pydantic_BaseModel):
    topics: list[str]
    decisions: list[str]
    action_items: list[str]
    sentiment: str
    word_count: int

def structured_summary(file_path: str) -> ConversationSummary:
    with open(file_path, 'r') as f:
        chat_history = f.read()
    
    client = ollama.Client()
    response = client.chat(
        model=MODELS['fast'],
        messages=[{
            "role": "user",
            "content": f"Analyze this conversation and output JSON with topics, decisions, action items, and sentiment:\n{chat_history}"
        }],
        format="json"
    )
    
    return ConversationSummary.model_validate_json(response['message']['content'])
