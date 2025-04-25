"""
DeepSeek LLM utilities for generating structured responses.
"""

import json
import re
from typing import Any, Dict

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from app.common.monitor import logger
from app.constants import DEEPSEEK_API_KEY

# Singleton instance of DeepSeek client
client: OpenAI | None = None


def load_client() -> OpenAI:
    """Get the DeepSeek client."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


def query_llm_json(
    prompt: str, temperature: float = 0, reasoning: bool = False
) -> Dict[str, Any]:
    global client
    if client is None:
        client = load_client()

    logger.log_llm_prompt(prompt)
    if reasoning:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        message = response.choices[0].message.content
        logger.log_llm_response(message)
        pattern = r"```json\n(.*)\n```"
        match = re.search(pattern, message, re.DOTALL)
        if match:
            response_json = json.loads(match.group(1))
            logger.log_llm_response(response_json)
            return response_json
        else:
            raise ValueError("No JSON found in the response")
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        response_json = json.loads(response.choices[0].message.content)
        logger.log_llm_response(response_json)
        return response_json


def query_llm_text(prompt: str, temperature: float = 0, reasoning: bool = False) -> str:
    global client
    if client is None:
        client = load_client()

    logger.log_llm_prompt(prompt)
    if reasoning:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    message = response.choices[0].message.content
    logger.log_llm_response(message)
    return message
