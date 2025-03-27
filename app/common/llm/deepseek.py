"""
DeepSeek LLM utilities for generating structured responses.
"""

import os
import sys
import json
from typing import Any, Dict, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_deepseek_llm(temperature: float = 0):
    """
    Get a configured instance of the DeepSeek LLM.
    
    Args:
        temperature: The temperature for generation (0-1)
        
    Returns:
        Configured ChatOpenAI instance
    """
    # Get API key from environment variables
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not found.")
        print("Please set it in your .env file or export it to your environment.")
        sys.exit(1)

    return ChatOpenAI(
        model="deepseek-chat",
        temperature=temperature,
        openai_api_key=deepseek_api_key,
        openai_api_base='https://api.deepseek.com'
    )


def query_llm_json(
    prompt: str, 
    json_schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0
) -> Dict[str, Any]:
    """
    Query the DeepSeek LLM with a prompt and get structured JSON output.
    
    Args:
        prompt: The user prompt to send to the LLM
        json_schema: Optional JSON schema to structure the response
        temperature: The temperature for generation (0-1)
        
    Returns:
        The structured JSON response
    """
    llm = get_deepseek_llm(temperature)
    
    if json_schema:
        # Use a system prompt that instructs the model to follow the schema
        system_prompt = f"""
        You are an AI assistant that generates responses in JSON format.
        Your response must adhere to the following JSON schema:
        {json.dumps(json_schema, indent=2)}
        
        Provide only valid JSON in your response. No explanations, no markdown formatting.
        """
        
        # Create the prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        # Create the chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Run the chain
        result = chain.invoke({"query": prompt})
    else:
        # Use a simpler system prompt for generic JSON
        system_prompt = """
        You are an AI assistant that generates responses in JSON format.
        Your JSON should be well-structured and contain relevant information.
        Provide only valid JSON in your response, with no additional text.
        No explanations, no markdown formatting.
        """
        
        # Create the prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        # Create the chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Run the chain
        result = chain.invoke({"query": prompt})
    
    # Parse the JSON result
    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return an error
        return {
            "error": "Failed to parse LLM output as JSON",
            "details": str(e),
            "raw_output": result
        } 