"""
DeepSeek LLM utilities for generating structured responses.
"""

import os
import sys
import json
from typing import Any, Dict, Optional, Union
from openai import OpenAI

from app.common.monitor import logger

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    logger.log_error("DEEPSEEK_API_KEY environment variable not found.")
    logger.log_error("Please set it in your .env file or export it to your environment.")
    sys.exit(1)

client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com",
)

def query_llm_json(prompt: str, temperature: float = 0) -> Dict[str, Any]:
    logger.log_llm_prompt(prompt)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    response_json = json.loads(response.choices[0].message.content)
    logger.log_llm_response(response_json)
    return response_json