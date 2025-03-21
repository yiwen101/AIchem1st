class QuestionEvaluator:
    def __init__(self, deepseek_model):
        """
        Initialize the QuestionEvaluator.
        
        Args:
            deepseek_model: The DeepSeek model instance
        """
        self.model = deepseek_model
    
    def evaluate(self, question, context, notebook_entries=None):
        """
        Evaluate a question to determine its type and required tools.
        
        Args:
            question (str): The question to evaluate
            context (str): Context knowledge
            notebook_entries (list, optional): Previous Q&A entries
            
        Returns:
            dict: Evaluation result with question type and required tools
        """
        return self.model.evaluate_question(question, context, notebook_entries) 