"""
DeepSeek LLM utilities for generating structured responses.
"""

import os
import sys
import json
from typing import Any, Dict, Optional, Union
from openai import OpenAI



deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    print("Error: DEEPSEEK_API_KEY environment variable not found.")
    print("Please set it in your .env file or export it to your environment.")
    sys.exit(1)

client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com",
)

def query_llm_json(prompt: str, temperature: float = 0) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)