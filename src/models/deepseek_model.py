import requests
import json
import os
from ..utils.config import Config

class DeepseekModel:
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.model_name = Config.MODEL_NAME
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
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
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def evaluate_question(self, question, context, notebook_entries=None):
        """
        Determine if a question is primitive or complex.
        
        Args:
            question (str): The question to evaluate
            context (str): Context knowledge
            notebook_entries (list, optional): Previous Q&A entries
            
        Returns:
            dict: The evaluation result including question type and reasoning
        """
        system_message = (
            "You are an expert at analyzing questions about videos. "
            "Determine if a question is primitive (can be answered directly) "
            "or complex (needs to be broken down into sub-questions)."
        )
        
        notebook_context = ""
        if notebook_entries:
            notebook_context = "Previous questions and answers:\n" + json.dumps(notebook_entries, indent=2)
        
        prompt = f"""
        Question: {question}
        
        Context Knowledge: {context}
        
        {notebook_context}
        
        Analyze this question and classify it as either:
        1. PRIMITIVE: Can be answered directly using the context, notebook, or a single tool
        2. COMPLEX: Needs to be broken down into sub-questions
        
        Return your answer in JSON format with:
        - "type": "PRIMITIVE" or "COMPLEX"
        - "reasoning": Explanation of your classification
        - "tools_needed": List of tools that might be needed (if any)
        """
        
        response = self.generate_response(prompt, system_message)
        
        try:
            # Try to parse the response as JSON
            return json.loads(response)
        except:
            # If parsing fails, return a default structure
            return {
                "type": "COMPLEX",
                "reasoning": "Failed to properly evaluate the question",
                "tools_needed": []
            }
    
    def generate_sub_question(self, main_question, context, notebook_entries=None):
        """
        Generate a relevant sub-question for a complex question.
        
        Args:
            main_question (str): The main complex question
            context (str): Context knowledge
            notebook_entries (list, optional): Previous Q&A entries
            
        Returns:
            str: A sub-question to help answer the main question
        """
        system_message = (
            "You are an expert at breaking down complex questions about videos "
            "into simpler sub-questions. Generate the most important next "
            "sub-question to ask that would help answer the main question."
        )
        
        notebook_context = ""
        if notebook_entries:
            notebook_context = "Previous questions and answers:\n" + json.dumps(notebook_entries, indent=2)
        
        prompt = f"""
        Main Question: {main_question}
        
        Context Knowledge: {context}
        
        {notebook_context}
        
        Generate a single, specific sub-question that would be most helpful in answering 
        the main question. The sub-question should:
        
        1. Start with what, why, how, when, who, where, etc.
        2. Be clearly related to the main question
        3. Be answerable with the available context or tools
        4. Help make progress toward answering the main question
        
        Return only the sub-question without any explanation or preamble.
        """
        
        return self.generate_response(prompt, system_message)
    
    def answer_primitive_question(self, question, context, notebook_entries=None, tool_results=None):
        """
        Answer a primitive question using available information.
        
        Args:
            question (str): The question to answer
            context (str): Context knowledge
            notebook_entries (list, optional): Previous Q&A entries
            tool_results (dict, optional): Results from tools
            
        Returns:
            str: The answer to the question
        """
        system_message = (
            "You are an expert at answering questions about videos. "
            "Provide a direct, accurate answer based on the available information."
        )
        
        notebook_context = ""
        if notebook_entries:
            notebook_context = "Previous questions and answers:\n" + json.dumps(notebook_entries, indent=2)
        
        tool_context = ""
        if tool_results:
            tool_context = "Tool results:\n" + json.dumps(tool_results, indent=2)
        
        prompt = f"""
        Question: {question}
        
        Context Knowledge: {context}
        
        {notebook_context}
        
        {tool_context}
        
        Please answer the question directly and concisely based on the available information.
        If you cannot answer with certainty, state what you can determine and what is unclear.
        """
        
        return self.generate_response(prompt, system_message)
    
    def combine_sub_answers(self, main_question, sub_questions_answers, context):
        """
        Combine answers to sub-questions to answer the main question.
        
        Args:
            main_question (str): The original complex question
            sub_questions_answers (list): Sub-questions and their answers
            context (str): Context knowledge
            
        Returns:
            str: Answer to the main question
        """
        system_message = (
            "You are an expert at synthesizing information to answer complex questions. "
            "Use the answers to sub-questions to provide a comprehensive answer to the main question."
        )
        
        sub_qa_formatted = "\n\n".join([
            f"Sub-question: {qa['question']}\nAnswer: {qa['answer']}" 
            for qa in sub_questions_answers
        ])
        
        prompt = f"""
        Main Question: {main_question}
        
        Context Knowledge: {context}
        
        Sub-questions and answers:
        {sub_qa_formatted}
        
        Based on these sub-question answers, provide a comprehensive answer to the main question.
        Your answer should synthesize the information from the sub-questions and be directly 
        relevant to what was asked in the main question.
        """
        
        return self.generate_response(prompt, system_message) 