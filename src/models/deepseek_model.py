import requests
import json
import os
import re
from utils.config import Config

class DeepseekModel:
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.model_name = Config.MODEL_NAME
        self.api_url = "https://api.deepseek.com/chat/completions"
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    def generate_response(self, prompt, system_message=None, temperature=0.7):
        """
        Generate a response from the DeepSeek model.
        
        Args:
            prompt (str): The user prompt
            system_message (str, optional): System instruction
            temperature (float, optional): Controls randomness
            
        Returns:
            str: The model's response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "deep_thinking":True,
            "max_tokens": 3000
        }
        print(f"Sending data to DeepSeek API: {data}")
        print("\n\n")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            resp = result["choices"][0]["message"]["content"]
            print(f"Received response from DeepSeek API: {resp}")
            print("\n\n")
            return resp
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _extract_json_from_response(self, response):
        """
        Extract JSON content from a response that might contain code fences or other text.
        
        Args:
            response (str): The response string
            
        Returns:
            str: The extracted JSON string
        """
        # Check if the response contains JSON code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, response)
        
        if matches:
            # Use the first JSON block found
            return matches[0].strip()
        
        # If no code blocks found, return the original response
        return response.strip()
        
    def determine_next_action(self, question, context, notebook_entries=None, tool_results=None, available_tools=None):
        """
        Unified method to determine the next action: direct answer, tool use, or sub-question.
        
        Args:
            question (str): The question to analyze
            context (str): Context knowledge
            notebook_entries (list, optional): Previous Q&A entries
            tool_results (dict, optional): Results from previously executed tools
            available_tools (dict, optional): Dictionary of available tools with their metadata
            
        Returns:
            dict: Action to take with the following structure:
                {
                    "action_type": "DIRECT_ANSWER" | "USE_TOOL" | "SUB_QUESTION",
                    "content": Answer text or sub-question text,
                    "tool_requests": List of tool requests if action_type is USE_TOOL,
                    "reasoning": Explanation of the decision
                }
        """
        system_message = (
            "You are an expert AI assistant that can answer questions and use tools. "
            "For each question, determine the most efficient way to answer it: "
            "1. DIRECT_ANSWER: If you already know the answer or can derive it from context/notebook "
            "2. USE_TOOL: If you need specific information from a tool "
            "3. SUB_QUESTION: If the question is complex and should be broken down first"
        )
        
        notebook_context = ""
        if notebook_entries:
            notebook_context = "Known information:\n" + json.dumps(notebook_entries, indent=2)
        
        tool_context = ""
        if tool_results:
            tool_context = "Tool results:\n" + json.dumps(tool_results, indent=2)
        
        tools_info = ""
        if available_tools:
            tools_info = "Available Tools:\n" + json.dumps(available_tools, indent=2)
        
        prompt = f"""
        Question: {question}
        
        Context Knowledge: {context}
        
        {notebook_context}
        
        {tool_context}
        
        {tools_info}
        
        Determine the most appropriate action to take for this question.
        Your response must be in valid JSON format with these fields:
        
        {{
          "action_type": "DIRECT_ANSWER" or "USE_TOOL" or "SUB_QUESTION",
          "reasoning": "Your explanation of why this action is needed",
          "content": "Either the direct answer or a specific sub-question",
          "tool_requests": [] // Only include and populate if action_type is USE_TOOL
        }}
        
        For USE_TOOL, include "tool_requests" as an array with this format:
        [
          {{
            "tool": "ExactToolName",
            "params": {{
              "param1": "value1",
              "param2": "value2"
            }}
          }}
        ]
        
        Return only the JSON object with no other text.
        """
        
        response = self.generate_response(prompt, system_message)
        print(f"Next action determination response: {response}")
        
        try:
            # Extract JSON from response and parse it
            json_str = self._extract_json_from_response(response)
            action_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["action_type", "content", "reasoning"]
            for field in required_fields:
                if field not in action_data:
                    action_data[field] = "" if field == "content" else "unknown"
            
            # Ensure action_type is valid
            valid_actions = ["DIRECT_ANSWER", "USE_TOOL", "SUB_QUESTION"]
            if action_data["action_type"] not in valid_actions:
                action_data["action_type"] = "SUB_QUESTION"
            
            # Validate tool requests if present
            if action_data["action_type"] == "USE_TOOL":
                if "tool_requests" not in action_data or not isinstance(action_data["tool_requests"], list):
                    action_data["tool_requests"] = []
                
                # Filter valid tool requests
                if available_tools:
                    valid_requests = []
                    for req in action_data["tool_requests"]:
                        if not isinstance(req, dict) or "tool" not in req:
                            continue
                            
                        if req["tool"] not in available_tools:
                            print(f"Warning: Tool '{req['tool']}' not found in available tools")
                            continue
                            
                        if "params" not in req or not isinstance(req["params"], dict):
                            req["params"] = {}
                            
                        valid_requests.append(req)
                    
                    action_data["tool_requests"] = valid_requests
            
            return action_data
            
        except Exception as e:
            # If parsing fails, return a default structure
            print(f"Error parsing action determination: {str(e)}")
            print(f"Response was: {response}")
            try:
                # Make a second attempt by removing backticks or other markdown manually
                clean_response = response.replace("```json", "").replace("```", "").strip()
                action_data = json.loads(clean_response)
                return action_data
            except:
                # If all parsing attempts fail, return default structure
                return {
                    "action_type": "DIRECT_ANSWER",
                    "content": "I couldn't process this question properly. Please try asking in a different way.",
                    "reasoning": "Failed to parse the model's response",
                    "tool_requests": []
                } 