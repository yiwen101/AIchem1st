class AgentCoordinator:
    def __init__(
        self, 
        question_solver, 
        notebook_manager,
        max_recursion_depth=5
    ):
        """
        Initialize the AgentCoordinator.
        
        Args:
            question_solver: The QuestionSolver instance
            notebook_manager: The NotebookManager instance
            max_recursion_depth (int, optional): Maximum recursion depth
        """
        self.question_solver = question_solver
        self.notebook_manager = notebook_manager
        self.max_recursion_depth = max_recursion_depth
    
    async def process_query(self, query, context_knowledge, parent_id=None, depth=0, prev_tool_results=None, query_entry_id=None):
        """
        Process a user query or sub-question using a unified approach.
        
        Args:
            query (str): The question to process
            context_knowledge (str): Context information
            parent_id (str, optional): ID of the parent question
            depth (int, optional): Current recursion depth
            prev_tool_results (dict, optional): Results from previously executed tools
            query_entry_id (str, optional): ID of an existing notebook entry for this query
            
        Returns:
            dict: Result with answer and entry_id
        """
        # Check recursion depth
        if depth >= self.max_recursion_depth:
            answer = "Maximum recursion depth reached. The question is too complex to answer fully."
            entry_id = query_entry_id or self.notebook_manager.add_entry(query, answer, parent_id=parent_id, tool_results={})
            if query_entry_id:
                # Update existing entry
                entry = self.notebook_manager.get_entry_by_id(entry_id)
                if entry:
                    entry["answer"] = answer
                    self.notebook_manager._save_notebook()
            return {"answer": answer, "entry_id": entry_id}
        
        try:
            # Get recent entries from notebook for context
            recent_entries = self.notebook_manager.get_recent_entries()
            
            # Get tool metadata
            available_tools = {
                name: meta for name, meta in 
                self.question_solver.tool_manager.get_tool_metadata().items()
            }
            
            # Determine next action
            next_action = self.question_solver.model.determine_next_action(
                query, 
                context_knowledge, 
                notebook_entries=recent_entries,
                tool_results=prev_tool_results,
                available_tools=available_tools
            )
            
            print(f"Next action: {next_action['action_type']} - {next_action['reasoning']}")
            
            # Create or get entry ID for this query
            tools_used = []
            placeholder_answer = "This question is not answered yet"
            
            if query_entry_id:
                # Use existing entry
                entry_id = query_entry_id
                # Get existing tools used
                entry = self.notebook_manager.get_entry_by_id(entry_id)
                if entry and "tools_used" in entry:
                    tools_used = entry["tools_used"]
            else:
                # Create a new entry with a placeholder answer
                entry_id = self.notebook_manager.add_entry(
                    query, 
                    placeholder_answer, 
                    tools_used=tools_used,
                    tool_results={},
                    parent_id=parent_id
                )
            
            # Process based on the determined action
            if next_action["action_type"] == "DIRECT_ANSWER":
                # Direct answer case
                answer = next_action["content"]
                
                # Update the notebook entry
                entry = self.notebook_manager.get_entry_by_id(entry_id)
                if entry:
                    entry["answer"] = answer
                    self.notebook_manager._save_notebook()
                
                return {"answer": answer, "entry_id": entry_id}
                
            elif next_action["action_type"] == "USE_TOOL":
                # Tool use case
                tool_requests = next_action.get("tool_requests", [])
                
                if tool_requests:
                    # Execute the tools
                    print(f"Executing tools: {[req.get('tool') for req in tool_requests]}")
                    tool_results = self.question_solver.execute_tool_requests(tool_requests)
                    current_tools_used = [req.get("tool") for req in tool_requests if req.get("tool")]
                    
                    # Update tools used in the entry
                    tools_used.extend(current_tools_used)
                    entry = self.notebook_manager.get_entry_by_id(entry_id)
                    if entry:
                        entry["tools_used"] = tools_used
                        # Store tool results in the entry
                        if "tool_results" not in entry:
                            entry["tool_results"] = {}
                        for tool_name, result in tool_results.items():
                            entry["tool_results"][tool_name] = result
                        self.notebook_manager._save_notebook()
                    
                    # Process the query again with tool results
                    return await self.process_query(
                        query,
                        context_knowledge,
                        parent_id=parent_id,
                        depth=depth + 1,
                        prev_tool_results=tool_results,
                        query_entry_id=entry_id
                    )
                else:
                    # No valid tool requests, treat as direct answer
                    answer = next_action["content"] or "I couldn't determine which tools to use for this question."
                    
                    # Update the entry
                    entry = self.notebook_manager.get_entry_by_id(entry_id)
                    if entry:
                        entry["answer"] = answer
                        self.notebook_manager._save_notebook()
                    
                    return {"answer": answer, "entry_id": entry_id}
                
            else:  # SUB_QUESTION
                # Sub-question case
                sub_question = next_action["content"]
                
                if not sub_question:
                    # If no sub-question provided, generate a default one
                    sub_question = f"What specific aspect of '{query}' should I investigate first?"
                
                print(f"Investigating sub-question: {sub_question}")
                
                # Process the sub-question recursively
                sub_result = await self.process_query(
                    sub_question, 
                    context_knowledge, 
                    parent_id=entry_id,  # Use current entry as parent for sub-questions
                    depth=depth + 1
                )
                
                # Get all sub-questions and answers for this question
                entry = self.notebook_manager.get_entry_by_id(entry_id)
                sub_questions_answers = entry.get("sub_questions", []) if entry else []
                
                # Add the latest sub-question if not already included
                sub_entry = self.notebook_manager.get_entry_by_id(sub_result["entry_id"])
                if sub_entry:
                    # Check if this sub-question is already in the list
                    is_duplicate = False
                    for existing_sub in sub_questions_answers:
                        if existing_sub.get("id") == sub_entry["id"]:
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        sub_questions_answers.append({
                            "id": sub_entry["id"],
                            "question": sub_entry["question"],
                            "answer": sub_entry["answer"]
                        })
                
                # If we have enough sub-questions or at a certain depth, try to answer the main question
                if len(sub_questions_answers) >= 2 or depth >= self.max_recursion_depth - 1:
                    # Generate a final answer based on the sub-question results
                    next_action = self.question_solver.model.determine_next_action(
                        query, 
                        context_knowledge, 
                        notebook_entries=sub_questions_answers,
                        available_tools=available_tools
                    )
                    
                    # Use the content as the final answer
                    final_answer = next_action["content"]
                    
                    if not final_answer:
                        final_answer = "Based on the investigations, I couldn't form a complete answer to your question."
                    
                    # Update the entry with the final answer
                    entry = self.notebook_manager.get_entry_by_id(entry_id)
                    if entry:
                        entry["answer"] = final_answer
                        self.notebook_manager._save_notebook()
                    
                    return {"answer": final_answer, "entry_id": entry_id}
                else:
                    # Generate another sub-question for further exploration
                    return await self.process_query(
                        query, 
                        context_knowledge, 
                        parent_id=parent_id,
                        depth=depth,
                        query_entry_id=entry_id
                    )
        except Exception as e:
            # Handle any unexpected errors during processing
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            
            # Update or create entry with error message
            if query_entry_id:
                entry = self.notebook_manager.get_entry_by_id(query_entry_id)
                if entry:
                    entry["answer"] = error_message
                    self.notebook_manager._save_notebook()
                    entry_id = query_entry_id
                else:
                    entry_id = self.notebook_manager.add_entry(query, error_message, parent_id=parent_id, tool_results={})
            else:
                entry_id = self.notebook_manager.add_entry(query, error_message, parent_id=parent_id, tool_results={})
                
            return {"answer": error_message, "entry_id": entry_id} 