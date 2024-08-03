from litellm import completion
from config import cfg


def llm(prompt):
    response = completion(
        messages=[{"content": prompt, "role": "user"}],
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        model=cfg.get("model"),
        custom_llm_provider="openai",
    )
    return response.choices[0].message.content