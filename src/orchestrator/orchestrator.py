from typing import Dict, List, Any, Optional
import time

from src.core.knowledge_manager import KnowledgeManager
from src.core.planner import Planner
from src.executor.executor import Executor
from src.adapter.llm_adapter import LLMAdapter
from src.models.interfaces import Plan, Action, ActionStatus, Tool, Entity
from src.utils.config import Config
from src.utils.logging import LoggingManager
from src.tools import TerminationSignal

logger = LoggingManager.get_logger()


class Orchestrator:
    """
    Main orchestrator for the tool planning agent.
    Manages the execution flow and coordinates between components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the orchestrator with components.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Initialize configuration
        self.config = Config(config_path)
        
        # Configure logging
        log_level = self.config.get("logging_level", "INFO")
        LoggingManager.configure(level=log_level)
        
        # Initialize components
        self.llm_adapter = LLMAdapter(self.config)
        self.knowledge_manager = KnowledgeManager()
        self.planner = Planner(self.llm_adapter)
        self.executor = Executor()
        
        # Internal state
        self.current_plan: Optional[Plan] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.current_phase_history: List[Dict[str, Any]] = []
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool for use by the agent.
        
        Args:
            tool: The tool to register
        """
        self.executor.register_tool(tool)
    
    def register_tools(self, tools: List[Tool]) -> None:
        """
        Register multiple tools for use by the agent.
        
        Args:
            tools: List of tools to register
        """
        self.executor.register_tools(tools)
    
    def initialize(self, goal: str, initial_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the agent with a goal and optional initial context.
        
        Args:
            goal: The goal to achieve
            initial_context: Optional initial context
        """
        logger.info(f"Initializing agent with goal: {goal}")
        
        # Clear previous state
        self.current_plan = None
        self.execution_history = []
        self.current_phase_history = []
        self.knowledge_manager.clear()
        
        # Initialize context if provided
        if initial_context:
            self.knowledge_manager.update_from_dict(initial_context)
        
        # Create initial plan
        available_tools = self.executor.list_tools()
        context = self.knowledge_manager.get_context_snapshot()
        self.current_plan = self.planner.create_plan(goal, available_tools, context)
    
    def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute a single cycle of the agent.
        A cycle consists of:
        1. Generating actions for the current phase/subgoal
        2. Executing those actions
        3. Updating knowledge with the results
        4. Evaluating if the phase is complete
        5. Potentially revising the plan
        
        Returns:
            Status information about the execution cycle
        """
        if not self.current_plan:
            error_msg = "No current plan. Call initialize() first."
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        if self.current_plan.current_phase >= len(self.current_plan.phases):
            logger.info("Plan execution complete")
            return {"status": "complete", "message": "Plan execution complete"}
        
        # Get current phase
        current_phase = self.current_plan.phases[self.current_plan.current_phase]
        logger.info(f"Executing cycle for phase {self.current_plan.current_phase + 1}/{len(self.current_plan.phases)}: {current_phase.name}")
        
        try:
            # 1. Generate next actions for the current phase
            available_tools = self.executor.list_tools()
            context = self.knowledge_manager.get_context_snapshot()
            next_actions = self.planner.generate_next_actions(
                self.current_plan, 
                context, 
                available_tools, 
                self.current_phase_history
            )
            
            if not next_actions:
                logger.info(f"No actions generated for phase {current_phase.name}")
                # If no actions were generated but we have previous actions, evaluate if phase is complete
                if self.current_phase_history:
                    phase_complete = self.planner.evaluate_phase_completion(
                        self.current_plan, 
                        self.current_phase_history
                    )
                    
                    if phase_complete:
                        logger.info(f"Phase {current_phase.name} is complete. Moving to next phase.")
                        current_phase.completed = True
                        self.current_plan.current_phase += 1
                        self.current_phase_history = []  # Reset for next phase
                        
                        if self.current_plan.current_phase >= len(self.current_plan.phases):
                            logger.info("All phases complete. Plan execution finished.")
                            return {"status": "complete", "message": "Plan execution complete"}
                        
                        return {
                            "status": "phase_complete",
                            "message": f"Phase '{current_phase.name}' completed. Moving to next phase.",
                            "next_phase": self.current_plan.phases[self.current_plan.current_phase].name
                        }
                    else:
                        # Revise the plan since we're stuck
                        logger.info("No actions and phase not complete. Revising plan.")
                        self.current_plan = self.planner.revise_plan(
                            self.current_plan,
                            self.current_phase_history[-5:] if len(self.current_phase_history) > 5 else self.current_phase_history,
                            context
                        )
                        return {
                            "status": "plan_revised",
                            "message": "Plan revised due to no progress on current phase."
                        }
                else:
                    # No actions and no history - something's wrong
                    logger.warning("No actions generated and no previous actions. Revising plan.")
                    self.current_plan = self.planner.revise_plan(
                        self.current_plan,
                        [],
                        context
                    )
                    return {
                        "status": "plan_revised",
                        "message": "Plan revised due to no actions generated."
                    }
            
            # 2. Execute the generated actions
            logger.info(f"Executing {len(next_actions)} actions")
            executed_actions = self.executor.execute_actions(next_actions)
            
            # 3. Update knowledge with action results
            for action in executed_actions:
                if action.status == ActionStatus.COMPLETED and action.result:
                    self._update_knowledge_from_action(action)
            
            # 4. Add to execution history and current phase history
            for action in executed_actions:
                action_record = {
                    "phase": current_phase.name,
                    "tool": action.tool,
                    "purpose": action.purpose,
                    "params": action.params,
                    "status": action.status.value,
                    "result": action.result,
                    "error": action.error,
                    "timestamp": time.time()
                }
                self.execution_history.append(action_record)
                self.current_phase_history.append(action_record)
            
            # 5. Evaluate if the phase is complete
            phase_complete = self.planner.evaluate_phase_completion(
                self.current_plan, 
                self.current_phase_history
            )
            
            if phase_complete:
                logger.info(f"Phase {current_phase.name} is complete. Moving to next phase.")
                current_phase.completed = True
                self.current_plan.current_phase += 1
                self.current_phase_history = []  # Reset for next phase
                
                if self.current_plan.current_phase >= len(self.current_plan.phases):
                    logger.info("All phases complete. Plan execution finished.")
                    return {"status": "complete", "message": "Plan execution complete"}
                
                return {
                    "status": "phase_complete",
                    "message": f"Phase '{current_phase.name}' completed. Moving to next phase.",
                    "next_phase": self.current_plan.phases[self.current_plan.current_phase].name
                }
            
            # 6. Periodically revise the plan based on accumulated observations
            if len(self.current_phase_history) > 0 and len(self.current_phase_history) % 5 == 0:
                logger.info("Periodic plan revision check")
                new_plan = self.planner.revise_plan(
                    self.current_plan,
                    self.current_phase_history[-5:],
                    context
                )
                
                # Only update if the plan actually changed
                if len(new_plan.phases) != len(self.current_plan.phases) or any(
                    new_phase.name != old_phase.name 
                    for new_phase, old_phase in zip(new_plan.phases[self.current_plan.current_phase:], 
                                                   self.current_plan.phases[self.current_plan.current_phase:])
                ):
                    logger.info("Plan revised based on new observations")
                    self.current_plan = new_plan
                    return {
                        "status": "plan_revised",
                        "message": "Plan revised based on execution results"
                    }
            
            # Return status for this cycle
            return {
                "status": "in_progress",
                "phase": current_phase.name,
                "actions_executed": len(executed_actions),
                "actions_succeeded": sum(1 for a in executed_actions if a.status == ActionStatus.COMPLETED),
                "actions_failed": sum(1 for a in executed_actions if a.status == ActionStatus.FAILED)
            }
            
        except TerminationSignal as e:
            # Handle termination signal
            logger.info(f"Execution terminated: {str(e)}")
            
            # Get the termination reason and result
            termination_reason = e.reason
            termination_result = getattr(e, 'result', None)
            
            # Add termination to execution history
            termination_record = {
                "phase": current_phase.name,
                "tool": "terminate",
                "purpose": "User-requested termination",
                "status": "completed",
                "result": {
                    "reason": termination_reason,
                    "has_result": termination_result is not None
                },
                "error": None,
                "timestamp": time.time()
            }
            self.execution_history.append(termination_record)
            self.current_phase_history.append(termination_record)
            
            return {
                "status": "terminated",
                "message": f"Execution terminated: {termination_reason}",
                "phase": current_phase.name,
                "result": termination_result
            }
    
    def execute_plan(self, max_cycles: int = 10) -> Dict[str, Any]:
        """
        Execute the current plan until completion or max_cycles is reached.
        
        Args:
            max_cycles: Maximum number of execution cycles
            
        Returns:
            Final status information
        """
        if not self.current_plan:
            error_msg = "No current plan. Call initialize() first."
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        logger.info(f"Executing plan with up to {max_cycles} cycles")
        
        cycle_count = 0
        last_status = {"status": "starting"}
        
        while cycle_count < max_cycles:
            cycle_count += 1
            logger.info(f"Execution cycle {cycle_count}/{max_cycles}")
            
            try:
                status = self.execute_cycle()
                last_status = status
                
                if status["status"] in ["complete", "error", "terminated"]:
                    break
                    
            except TerminationSignal as e:
                # Handle termination at the top level in case it wasn't caught elsewhere
                logger.info(f"Execution terminated: {str(e)}")
                termination_result = getattr(e, 'result', None)
                last_status = {
                    "status": "terminated",
                    "message": f"Execution terminated: {str(e)}",
                    "result": termination_result
                }
                break
        
        if cycle_count >= max_cycles and last_status["status"] not in ["complete", "terminated", "error"]:
            logger.warning(f"Reached maximum cycles ({max_cycles}) without completing the plan")
            return {
                "status": "max_cycles_reached",
                "message": f"Plan execution stopped after {max_cycles} cycles",
                "execution_history": self.execution_history
            }
        
        return {
            "status": last_status["status"],
            "message": last_status.get("message", "Plan execution finished"),
            "execution_history": self.execution_history,
            "result": last_status.get("result")
        }
    
    def _update_knowledge_from_action(self, action: Action) -> None:
        """
        Update knowledge from action results.
        This is a basic implementation that will be enhanced in later phases.
        
        Args:
            action: The completed action
        """
        # Add the tool as an entity
        tool_entity_id = f"tool:{action.tool}"
        tool_entity = Entity(
            id=tool_entity_id,
            type="tool",
            properties={
                "name": action.tool,
                "purpose": action.purpose
            }
        )
        self.knowledge_manager.add_entity(tool_entity)
        
        # Add the result as an entity
        if isinstance(action.result, dict):
            # For dictionary results, add each key-value pair
            for key, value in action.result.items():
                result_entity_id = f"result:{action.tool}:{key}"
                result_entity = Entity(
                    id=result_entity_id,
                    type="result",
                    properties={
                        "tool": action.tool,
                        "key": key,
                        "value": str(value)
                    }
                )
                self.knowledge_manager.add_entity(result_entity)
                
                # Add relationship between tool and result
                self.knowledge_manager.add_relationship_if_entities_exist(
                    source=tool_entity_id,
                    target=result_entity_id,
                    rel_type="produced",
                    properties={
                        "timestamp": time.time()
                    }
                )
        else:
            # For non-dictionary results, add a single result entity
            result_entity_id = f"result:{action.tool}"
            result_entity = Entity(
                id=result_entity_id,
                type="result",
                properties={
                    "tool": action.tool,
                    "value": str(action.result)
                }
            )
            self.knowledge_manager.add_entity(result_entity)
            
            # Add relationship between tool and result
            self.knowledge_manager.add_relationship_if_entities_exist(
                source=tool_entity_id,
                target=result_entity_id,
                rel_type="produced",
                properties={
                    "timestamp": time.time()
                }
            ) 