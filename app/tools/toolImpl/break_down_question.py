"""
Question breakdown tool for complex video analysis queries.

This module provides a tool for breaking down complex questions into simpler sub-questions
that can be more easily answered by other tools.
"""

import os
from typing import Dict, Any, List, Optional

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.common.monitor import logger
from app.common.llm.openai import query_llm_text
from app.common.resource_manager.resource_manager import resource_manager

@register_tool
class BreakDownQuestionTool(BaseTool):
    """Tool for breaking down complex questions into simpler sub-questions."""
    
    name = "break_down_question"
    description = "Break down the current question into simpler sub-questions that can be answered individually."
    parameters = [
        ToolParameter(
            name="model",
            type=ToolParameterType.STRING,
            description="The model to use for breaking down the question",
            required=False,
            default="gpt-4o-mini"
        )
    ]
    
    @classmethod
    def execute(cls, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Break down the current question into simpler sub-questions.
        
        Args:
            model: The model to use for breaking down the question
            
        Returns:
            Dictionary with the list of sub-questions and reasoning
        """
        # Get the current question from the resource manager
        current_query = resource_manager.get_current_query()
        if not current_query:
            error_msg = "No current query set in resource manager"
            logger.log_error(error_msg)
            return {
                "error": error_msg,
                "sub_questions": [],
                "explanation": "No question to break down",
                "approach": ""
            }
        
        query = current_query.question
        logger.log_info(f"Breaking down question: {query}")
        
        prompt = f"""
        I'm analyzing a video and need to break down the following complex question into simpler sub-questions that can be answered individually:

        "{query}"

        Please break this down into:
        1. A list of 2-5 simpler, focused sub-questions that collectively help answer the main question
        2. A brief explanation of how these sub-questions contribute to answering the main question
        3. A suggested approach for tackling these questions (which video analysis tools might be useful)

        Format your response as a JSON object with these keys:
        - sub_questions: array of strings (the simpler questions)
        - explanation: string (how they contribute to the main question)
        - approach: string (suggested tools and techniques)
        """
        
        try:
            # Query the LLM to break down the question
            response = query_llm_text(prompt, model=model)
            
            # Parse the response
            # Note: The response might not be perfectly formatted JSON, 
            # but we'll try to extract the key components
            import json
            import re
            
            # Try to parse as JSON first
            try:
                parsed_response = json.loads(response)
                sub_questions = parsed_response.get("sub_questions", [])
                explanation = parsed_response.get("explanation", "")
                approach = parsed_response.get("approach", "")
            except json.JSONDecodeError:
                # If JSON parsing fails, extract the information using regex
                logger.log_warning(f"Could not parse response as JSON, falling back to regex extraction")
                
                # Extract sub-questions
                sub_questions_match = re.search(r'sub_questions"?\s*:?\s*\[(.*?)\]', response, re.DOTALL)
                if sub_questions_match:
                    sub_questions_text = sub_questions_match.group(1)
                    sub_questions = [q.strip(' "\'') for q in re.findall(r'"([^"]*)"', sub_questions_text)]
                    if not sub_questions:
                        sub_questions = [q.strip() for q in sub_questions_text.split(',') if q.strip()]
                else:
                    # Look for numbered questions
                    sub_questions = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', response, re.DOTALL)
                    sub_questions = [q.strip() for q in sub_questions if q.strip()]
                
                # Extract explanation
                explanation_match = re.search(r'explanation"?\s*:?\s*"(.*?)"(?:,|$)', response, re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    # Look for paragraphs that might contain the explanation
                    explanation = ""
                    for para in response.split('\n\n'):
                        if 'contribute' in para.lower() or 'help answer' in para.lower():
                            explanation = para.strip()
                            break
                
                # Extract approach
                approach_match = re.search(r'approach"?\s*:?\s*"(.*?)"(?:,|$)', response, re.DOTALL)
                if approach_match:
                    approach = approach_match.group(1).strip()
                else:
                    # Look for paragraphs that might contain the approach
                    approach = ""
                    for para in response.split('\n\n'):
                        if 'tool' in para.lower() or 'technique' in para.lower() or 'approach' in para.lower():
                            approach = para.strip()
                            break
            
            # Ensure we have at least some sub-questions
            if not sub_questions:
                sub_questions = ["Identify key objects in the video", 
                                "Determine the sequence of events",
                                "Analyze specific elements relevant to the query"]
                explanation = "Could not extract sub-questions from the response."
            
            result = {
                "query": query,
                "sub_questions": sub_questions,
                "explanation": explanation,
                "approach": approach,
                "raw_response": response
            }
            
            logger.log_info(f"Question breakdown yielded {len(sub_questions)} sub-questions")
            
            return result
        
        except Exception as e:
            logger.log_error(f"Error breaking down question: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "sub_questions": [],
                "explanation": f"Error occurred: {str(e)}",
                "approach": ""
            }
