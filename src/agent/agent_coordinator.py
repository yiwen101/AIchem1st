class AgentCoordinator:
    def __init__(
        self, 
        question_evaluator, 
        sub_question_generator, 
        question_solver, 
        notebook_manager,
        max_recursion_depth=5
    ):
        """
        Initialize the AgentCoordinator.
        
        Args:
            question_evaluator: The QuestionEvaluator instance
            sub_question_generator: The SubQuestionGenerator instance
            question_solver: The QuestionSolver instance
            notebook_manager: The NotebookManager instance
            max_recursion_depth (int, optional): Maximum recursion depth
        """
        self.question_evaluator = question_evaluator
        self.sub_question_generator = sub_question_generator
        self.question_solver = question_solver
        self.notebook_manager = notebook_manager
        self.max_recursion_depth = max_recursion_depth
    
    async def process_query(self, query, context_knowledge, parent_id=None, depth=0):
        """
        Process a user query or sub-question.
        
        Args:
            query (str): The question to process
            context_knowledge (str): Context information
            parent_id (str, optional): ID of the parent question
            depth (int, optional): Current recursion depth
            
        Returns:
            dict: Result with answer and entry_id
        """
        # Check recursion depth
        if depth >= self.max_recursion_depth:
            answer = "Maximum recursion depth reached. The question is too complex."
            entry_id = self.notebook_manager.add_entry(query, answer, parent_id=parent_id)
            return {"answer": answer, "entry_id": entry_id}
        
        # Get recent entries from notebook for context
        recent_entries = self.notebook_manager.get_recent_entries()
        
        # Evaluate the question
        evaluation = self.question_evaluator.evaluate(query, context_knowledge, recent_entries)
        
        # Process based on question type
        if evaluation["type"] == "PRIMITIVE":
            # Handle primitive question
            result = self.question_solver.solve_primitive(
                query, 
                context_knowledge, 
                recent_entries,
                evaluation.get("tools_needed", [])
            )
            
            # Save to notebook
            entry_id = self.notebook_manager.add_entry(
                query, 
                result["answer"], 
                tools_used=result["tools_used"],
                parent_id=parent_id
            )
            
            return {"answer": result["answer"], "entry_id": entry_id}
        else:
            # Handle complex question
            sub_question = self.sub_question_generator.generate(query, context_knowledge, recent_entries)
            
            # Process the sub-question recursively
            sub_result = await self.process_query(
                sub_question, 
                context_knowledge, 
                parent_id=parent_id,  # Initially use same parent
                depth=depth + 1
            )
            
            # Get all sub-questions and answers for this question
            entry = self.notebook_manager.get_entry_by_id(parent_id) if parent_id else None
            sub_questions_answers = entry.get("sub_questions", []) if entry else []
            
            # Add the latest sub-question
            sub_entry = self.notebook_manager.get_entry_by_id(sub_result["entry_id"])
            if sub_entry and sub_entry not in sub_questions_answers:
                sub_questions_answers.append({
                    "question": sub_entry["question"],
                    "answer": sub_entry["answer"]
                })
            
            # If we have enough sub-questions or at a certain depth, try to answer the main question
            if len(sub_questions_answers) >= 4 or depth >= self.max_recursion_depth - 1:
                final_answer = self.question_solver.combine_sub_answers(
                    query, 
                    sub_questions_answers, 
                    context_knowledge
                )
                
                # Save the final answer to notebook
                if parent_id:
                    # Update the parent entry with the final answer
                    parent_entry = self.notebook_manager.get_entry_by_id(parent_id)
                    if parent_entry:
                        parent_entry["answer"] = final_answer
                        self.notebook_manager._save_notebook()
                        return {"answer": final_answer, "entry_id": parent_id}
                
                # If there's no parent or parent not found, create a new entry
                entry_id = self.notebook_manager.add_entry(query, final_answer)
                return {"answer": final_answer, "entry_id": entry_id}
            else:
                # Generate another sub-question for further exploration
                return await self.process_query(
                    query, 
                    context_knowledge, 
                    parent_id=parent_id,
                    depth=depth
                ) 