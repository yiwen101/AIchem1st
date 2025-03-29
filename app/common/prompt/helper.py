"""
Prompt helper for generating formatted prompts with various contextual information.
"""

import json
from typing import Any, Dict, List, Optional, Union

from app.persistence.notebook.notebook_manager import load_notebook
from app.persistence.tool_call.tool_call_manager import load_tool_calls
from app.tools.tool_manager import format_tools_for_prompt

def format_notebook_info(qa_notebook: List[Dict[str, Any]]) -> str:
    """
    Format QA records from the state's qa_notebook into a readable prompt section.
    
    Args:
        qa_notebook: List of QA records from the state
        
    Returns:
        A formatted string with notebook information
    """
    if not qa_notebook:
        return "No previous questions and answers found."
    
    formatted = "# Previous Questions and Answers\n\n"
    
    for i, record in enumerate(qa_notebook, 1):
        formatted += f"## Question {i}: {record['question']}\n"
        
        if record.get('answer'):
            formatted += f"**Answer**: {record['answer']}\n"
            
            if record.get('reason'):
                formatted += f"**Reasoning**: {record['reason']}\n"
        else:
            formatted += "*This question has not been answered yet.*\n"
        
        formatted += "\n"
    
    return formatted + "\n\n"

def format_single_tool_call_info(tool_result: Any) -> str:
    """
    Format a single tool call result into a readable prompt section.
    """
    if isinstance(tool_result, (dict, list)):
        try:
            formatted = f"```json\n{json.dumps(tool_result, indent=2)}\n```\n"
        except TypeError:
            formatted = f"{str(tool_result)}\n"
    else:
        formatted = f"{str(tool_result)}\n"
    return formatted

def format_tool_call_info(tool_results: Dict[str, List[Any]]) -> str:
    """
    Format tool call information from the state into a readable prompt section.
    
    Args:
        tool_results: Tool results dictionary from the state
        
    Returns:
        A formatted string with tool call information
    """
    if not tool_results:
        return "No previous tool calls found."
    
    formatted = "# Previous Tool Calls\n\n"
    
    for tool_name, results in tool_results.items():
        formatted += f"## Tool: {tool_name}\n"
        
        if not results:
            formatted += "*No results for this tool.*\n\n"
            continue
        
        for i, result in enumerate(results, 1):
            formatted += f"### Result {i}\n"
            formatted += format_single_tool_call_info(result)
            
            formatted += "\n"
    
    return formatted + "\n\n"

def format_pre_question_info(state: Dict[str, Any]) -> str:
    """
    Format pre-question information from the state into a readable prompt section.
    """
    if "previous_QA" in state and state["previous_QA"]:
        formatted = "# Previous Q&A\n\n"
        prev_qa = state["previous_QA"]
        formatted += "**Previous Q&A**:\n"
        formatted += f"- Question: {prev_qa.get('question', 'N/A')}\n"
        formatted += f"- Answer: {prev_qa.get('answer', 'N/A')}\n"
        if prev_qa.get('reason'):
            formatted += f"- Reasoning: {prev_qa.get('reason', 'N/A')}\n"
        
        return formatted + "\n\n"
    return ""

def format_current_question_tool_results(state: Dict[str, Any]) -> str:
    """
    Format current question tool results from the state into a readable prompt section.
    """
    if "current_question_tool_results" in state and state["current_question_tool_results"]:
        formatted = "# Current Question Tool Results\n\n"
        for tool, result in state["current_question_tool_results"].items():
            formatted += f"## Tool: {tool}\n"
            formatted += format_single_tool_call_info(result)
        return formatted + "\n\n"
    return ""

def format_output_schema(schema: Optional[str]) -> str:
    """
    Format output schema information into a readable prompt section.
    
    Args:
        schema: JSON schema as a string or None
        
    Returns:
        A formatted string with schema information
    """
    if not schema:
        return ""
    
    formatted = "# Required Output Format\n\n"
    formatted += "Your response should adhere to the following JSON schema:\n\n"
    formatted += f"```json\n{schema}\n```\n\n"
    formatted += "Provide only valid JSON in your response.\n"
    
    return formatted + "\n\n"

def generate_prompt(
    prompt: str,
    state: Optional[Dict[str, Any]] = None,
    add_tool_info: bool = False,
    notebook_info: bool = False,
    tool_call_info: bool = False,
    pre_question_info: bool = False,
    current_question_tool_results: bool = False,
    output_schema: Optional[str] = None
) -> str:
    """
    Generate a complete prompt with various contextual information from the graph state.
    
    Args:
        prompt: The base user prompt
        state: The current state dictionary from the graph
        add_tool_info: Whether to include tool information from the state
        notebook_info: Whether to include notebook QA history
        tool_call_info: Whether to include tool call history
        pre_question_info: Whether to include previous question information
        current_question_tool_results: Whether to include current question tool results
        output_schema: Optional JSON schema for the expected output
        
    Returns:
        A complete formatted prompt
    """
    sections = [f"# User Query\n\n{prompt}\n\n"]

    if output_schema:
        sections.append(format_output_schema(output_schema))
    
    # Add state information if provided
    if pre_question_info and state:
        sections.append(format_pre_question_info(state))
        
    if current_question_tool_results and state:
        sections.append(format_current_question_tool_results(state))

    if notebook_info and state and "qa_notebook" in state:
        # Use notebook info from state
        sections.append(format_notebook_info(state["qa_notebook"]))
    
    # Add tool call information if requested
    if tool_call_info and state and "tool_results" in state:
        # Use tool call info from state
        sections.append(format_tool_call_info(state["tool_results"]))
    
    # Add tool information if requested
    if add_tool_info:
        sections.append(format_tools_for_prompt())
    
    # Combine all sections
    return "".join(sections) 