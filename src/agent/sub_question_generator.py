class SubQuestionGenerator:
    def __init__(self, deepseek_model):
        """
        Initialize the SubQuestionGenerator.
        
        Args:
            deepseek_model: The DeepSeek model instance
        """
        self.model = deepseek_model
    
    def generate(self, main_question, context, notebook_entries=None):
        """
        Generate a relevant sub-question for a complex question.
        
        Args:
            main_question (str): The main complex question
            context (str): Context knowledge
            notebook_entries (list, optional): Previous Q&A entries
            
        Returns:
            str: A sub-question to help answer the main question
        """
        return self.model.generate_sub_question(main_question, context, notebook_entries) 