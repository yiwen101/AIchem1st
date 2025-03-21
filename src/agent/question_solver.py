class QuestionSolver:
    def __init__(self, deepseek_model, tool_manager):
        """
        Initialize the QuestionSolver.
        
        Args:
            deepseek_model: The DeepSeek model instance
            tool_manager: The ToolManager instance
        """
        self.model = deepseek_model
        self.tool_manager = tool_manager
    
    def solve_primitive(self, question, context, notebook_entries=None, tools_needed=None):
        """
        Solve a primitive question using context, notebook, and tools as needed.
        
        Args:
            question (str): The question to solve
            context (str): Context knowledge
            notebook_entries (list, optional): Previous Q&A entries
            tools_needed (list, optional): List of tools that might be needed
            
        Returns:
            dict: The answer and any tools used
        """
        tool_results = None
        tools_used = []
        
        # If tools are needed, execute them
        if tools_needed and len(tools_needed) > 0:
            tool_requests = []
            
            for tool in tools_needed:
                # For now, we're just passing empty parameters
                # In a real implementation, we would extract parameters from the question
                tool_requests.append({
                    "tool": tool,
                    "params": {}
                })
            
            tool_results = self.tool_manager.execute_tools(tool_requests)
            tools_used = tools_needed
        
        # Get answer from model
        answer = self.model.answer_primitive_question(
            question, 
            context, 
            notebook_entries, 
            tool_results
        )
        
        return {
            "answer": answer,
            "tools_used": tools_used
        }
    
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
        return self.model.combine_sub_answers(main_question, sub_questions_answers, context) 