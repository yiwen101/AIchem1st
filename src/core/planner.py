from typing import Dict, List, Any, Optional

from src.models.interfaces import Plan, Phase, Action, Assumption, ContextNote, FallbackOption
from src.adapter.llm_adapter import LLMAdapter, PromptTemplate
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


class Planner:
    """
    Responsible for creating and revising execution plans.
    """
    
    # Template for plan generation
    PLAN_TEMPLATE = """
    Current Goal: {goal}
    
    Known Entities: 
    {entities}
    
    Available Tools:
    {tools}
    
    Generate a plan to achieve the goal using the available tools and considering the known entities.
    Your plan should divide the goal into subgoals (phases), each with clear success criteria.
    DO NOT include specific actions in the plan - we will generate actions dynamically as we work through each subgoal.
    Each phase should represent a meaningful step towards the main goal.
    
    Your plan should include:
    1. A series of phases/subgoals with clear objectives
    2. Success criteria for each phase/subgoal
    
    Think logically about the order of subgoals and dependencies between them.
    """
    
    # Template for generating next actions
    NEXT_ACTIONS_TEMPLATE = """
    Current Goal: {goal}
    
    Current Subgoal (Phase {phase_num}/{total_phases}): {phase_name}
    Description: {phase_description}
    Success Criteria: {success_criteria}
    
    Context Information:
    {context_info}
    
    Previous Actions and Results in this Phase:
    {previous_actions}
    
    Available Tools:
    {tools}
    
    Generate the next 1-3 actions to execute to make progress on the current subgoal.
    Be strategic and consider what we've already done and what information we've gathered.
    Consider the success criteria for this subgoal and what steps are needed to achieve it.
    
    For each action, specify:
    1. Which tool to use
    2. Exact parameters for the tool
    3. The purpose of this action and how it contributes to the subgoal
    4. Optional fallback actions in case this action fails
    
    If you believe the current subgoal has been achieved based on the success criteria and available information, 
    include a "terminate" action with a reason explaining why the subgoal is complete.
    """
    
    # Template for evaluating phase completion
    PHASE_EVALUATION_TEMPLATE = """
    Current Subgoal (Phase {phase_num}/{total_phases}): {phase_name}
    Description: {phase_description}
    Success Criteria: {success_criteria}
    
    Actions Taken and Results:
    {actions_results}
    
    Based on the success criteria and the results of actions taken, 
    evaluate whether this subgoal has been achieved.
    
    Considerations:
    - Have we gathered all the information needed for this subgoal?
    - Have we performed all necessary actions for this subgoal?
    - Do the results satisfy the success criteria?
    - Is there anything important missing that would prevent us from moving to the next subgoal?
    
    Provide your evaluation and reasoning.
    """
    
    # Template for plan revision
    PLAN_REVISION_TEMPLATE = """
    Current Plan: 
    {current_plan}
    
    Current Context Information:
    {context_info}
    
    Current Progress:
    - Completed phases: {completed_phases}
    - Current phase: {current_phase}
    - Remaining phases: {remaining_phases}
    
    Recent Observations:
    {recent_observations}
    
    Based on our progress and observations, revise the plan if necessary.
    You can:
    1. Keep the current plan if it's still valid
    2. Modify the remaining phases or add new phases
    3. Adjust success criteria based on new information
    4. Reorder remaining phases if needed
    
    Explain your reasoning for any changes.
    """
    
    def __init__(self, llm_adapter: LLMAdapter):
        """
        Initialize the planner.
        
        Args:
            llm_adapter: Adapter for LLM interaction
        """
        self.llm_adapter = llm_adapter
        self.plan_template = PromptTemplate(self.PLAN_TEMPLATE)
        self.next_actions_template = PromptTemplate(self.NEXT_ACTIONS_TEMPLATE)
        self.phase_evaluation_template = PromptTemplate(self.PHASE_EVALUATION_TEMPLATE)
        self.plan_revision_template = PromptTemplate(self.PLAN_REVISION_TEMPLATE)
    
    def create_plan(self, goal: str, available_tools: List[Dict[str, Any]], 
                   context: ContextNote) -> Plan:
        """
        Create an execution plan to achieve a goal.
        
        Args:
            goal: The goal to achieve
            available_tools: List of available tools
            context: Current context
            
        Returns:
            An execution plan
        """
        logger.info(f"Creating plan for goal: {goal}")
        
        # Format entity information
        entities_text = "\n".join([
            f"- {entity.type}: {entity.id} - {entity.properties.get('summary', 'No summary')}"
            for entity in context.entities.values()
        ]) or "No entities in context"
        
        # Format tool information
        tools_text = "\n".join([
            f"- {tool['name']}: {tool['description']} (Params: {tool['parameters']})"
            for tool in available_tools
        ])
        
        # Create the prompt
        prompt = self.plan_template.format(
            goal=goal,
            entities=entities_text,
            tools=tools_text
        )
        
        # Get plan from LLM
        plan_schema = {
            "type": "object",
            "properties": {
                "phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "success_criteria": {"type": "string"}
                        }
                    }
                },
                "assumptions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "confidence": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        logger.info("Generating plan using LLM")
        plan_data = self.llm_adapter.generate_json(prompt, plan_schema, deep_thinking=True)
        logger.info(f"Plan data: {plan_data}")
        # Convert JSON to plan objects
        phases = []
        for phase_data in plan_data.get("phases", []):
            # Create an empty list of actions - we'll generate these dynamically later
            phase = Phase(
                name=phase_data["name"],
                description=phase_data["description"],
                actions=[],  # Start with empty actions list
                success_criteria=phase_data["success_criteria"]
            )
            phases.append(phase)
        
        assumptions = []
        for assumption_data in plan_data.get("assumptions", []):
            assumption = Assumption(
                description=assumption_data["description"],
                confidence=assumption_data.get("confidence", 1.0)
            )
            assumptions.append(assumption)
        
        plan = Plan(goal=goal, phases=phases, assumptions=assumptions)
        logger.info(f"Created plan with {len(phases)} phases and {len(assumptions)} assumptions")
        
        return plan
    
    def generate_next_actions(self, plan: Plan, context: ContextNote, 
                             available_tools: List[Dict[str, Any]],
                             previous_actions: List[Dict[str, Any]] = None) -> List[Action]:
        """
        Generate the next set of actions for the current phase.
        
        Args:
            plan: The current plan
            context: Current context
            available_tools: List of available tools
            previous_actions: List of previous actions and their results in the current phase
            
        Returns:
            List of next actions to execute
        """
        if not plan.phases or plan.current_phase >= len(plan.phases):
            logger.warning("Cannot generate actions: invalid plan or phase")
            return []
        
        current_phase = plan.phases[plan.current_phase]
        logger.info(f"Generating actions for phase: {current_phase.name}")
        
        # Format context information
        context_items = []
        for entity in context.entities.values():
            props_str = ", ".join([f"{k}: {v}" for k, v in entity.properties.items()])
            context_items.append(f"- {entity.type}: {entity.id} - {props_str}")
        
        for rel in context.relationships:
            context_items.append(f"- Relationship: {rel.source} -> {rel.type} -> {rel.target}")
        
        context_info = "\n".join(context_items) or "No context information available"
        
        # Format previous actions and results
        prev_actions_text = ""
        if previous_actions:
            action_texts = []
            for idx, action_info in enumerate(previous_actions):
                action_text = f"{idx+1}. Tool: {action_info['tool']}, Purpose: {action_info['purpose']}"
                if action_info['status'] == "completed":
                    result_str = str(action_info['result']).replace('\n', ' ')[:100]
                    if len(str(action_info['result'])) > 100:
                        result_str += "..."
                    action_text += f"\n   Result: {result_str}"
                elif action_info['status'] == "failed":
                    action_text += f"\n   Failed: {action_info['error']}"
                action_texts.append(action_text)
            prev_actions_text = "\n".join(action_texts)
        else:
            prev_actions_text = "No previous actions in this phase"
        
        # Format tool information
        tools_text = "\n".join([
            f"- {tool['name']}: {tool['description']} (Params: {tool['parameters']})"
            for tool in available_tools
        ])
        
        # Create the prompt
        prompt = self.next_actions_template.format(
            goal=plan.goal,
            phase_num=plan.current_phase + 1,
            total_phases=len(plan.phases),
            phase_name=current_phase.name,
            phase_description=current_phase.description,
            success_criteria=current_phase.success_criteria,
            context_info=context_info,
            previous_actions=prev_actions_text,
            tools=tools_text
        )
        
        # Get next actions from LLM
        actions_schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "params": {"type": "object"},
                            "purpose": {"type": "string"},
                            "fallbacks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "tool": {"type": "string"},
                                        "params": {"type": "object"},
                                        "priority": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                },
                "reasoning": {"type": "string"}
            }
        }
        
        logger.info("Generating next actions using LLM")
        actions_data = self.llm_adapter.generate_json(prompt, actions_schema)
        
        # Log reasoning
        reasoning = actions_data.get("reasoning", "No reasoning provided")
        logger.info(f"Action generation reasoning: {reasoning}")
        
        # Convert JSON to action objects
        actions = []
        for action_data in actions_data.get("actions", []):
            fallbacks = []
            
            for fallback_data in action_data.get("fallbacks", []):
                fallback = FallbackOption(
                    tool=fallback_data["tool"],
                    params=fallback_data["params"],
                    priority=fallback_data.get("priority", 0)
                )
                fallbacks.append(fallback)
            
            action = Action(
                tool=action_data["tool"],
                params=action_data["params"],
                purpose=action_data["purpose"],
                fallbacks=fallbacks
            )
            actions.append(action)
        
        logger.info(f"Generated {len(actions)} next actions")
        return actions
    
    def evaluate_phase_completion(self, plan: Plan, actions_results: List[Dict[str, Any]]) -> bool:
        """
        Evaluate whether the current phase has been completed based on its success criteria.
        
        Args:
            plan: The current plan
            actions_results: List of actions and their results in the current phase
            
        Returns:
            True if the phase is complete, False otherwise
        """
        if not plan.phases or plan.current_phase >= len(plan.phases):
            logger.warning("Cannot evaluate phase: invalid plan or phase")
            return False
        
        current_phase = plan.phases[plan.current_phase]
        logger.info(f"Evaluating completion of phase: {current_phase.name}")
        
        # If no actions have been executed yet, the phase can't be complete
        if not actions_results:
            return False
        
        # Check for terminate actions - if any was successful, consider the phase complete
        for action in actions_results:
            if action['tool'] == 'terminate' and action['status'] == 'completed':
                logger.info(f"Phase marked as complete due to terminate action: {action['result']}")
                return True
        
        # Format actions and results
        actions_results_text = ""
        if actions_results:
            action_texts = []
            for idx, action_info in enumerate(actions_results):
                action_text = f"{idx+1}. Tool: {action_info['tool']}, Purpose: {action_info['purpose']}"
                if action_info['status'] == "completed":
                    result_str = str(action_info['result']).replace('\n', ' ')[:100]
                    if len(str(action_info['result'])) > 100:
                        result_str += "..."
                    action_text += f"\n   Result: {result_str}"
                elif action_info['status'] == "failed":
                    action_text += f"\n   Failed: {action_info['error']}"
                action_texts.append(action_text)
            actions_results_text = "\n".join(action_texts)
        else:
            actions_results_text = "No actions executed in this phase"
        
        # Create the prompt
        prompt = self.phase_evaluation_template.format(
            phase_num=plan.current_phase + 1,
            total_phases=len(plan.phases),
            phase_name=current_phase.name,
            phase_description=current_phase.description,
            success_criteria=current_phase.success_criteria,
            actions_results=actions_results_text
        )
        
        # Get evaluation from LLM
        evaluation_schema = {
            "type": "object",
            "properties": {
                "is_complete": {"type": "boolean"},
                "reasoning": {"type": "string"}
            }
        }
        
        logger.info("Evaluating phase completion using LLM")
        evaluation = self.llm_adapter.generate_json(prompt, evaluation_schema)
        
        is_complete = evaluation.get("is_complete", False)
        reasoning = evaluation.get("reasoning", "No reasoning provided")
        
        logger.info(f"Phase completion evaluation: {is_complete} - {reasoning}")
        return is_complete
    
    def revise_plan(self, plan: Plan, new_observations: List[Dict[str, Any]], 
                   context: ContextNote) -> Plan:
        """
        Revise an existing plan based on new observations.
        
        Args:
            plan: The current plan
            new_observations: New observations to consider
            context: Current context
            
        Returns:
            The revised plan
        """
        if not plan.phases:
            logger.warning("Cannot revise an empty plan")
            return plan
        
        logger.info(f"Revising plan for goal: {plan.goal}")
        
        # Format context information
        context_items = []
        for entity in context.entities.values():
            props_str = ", ".join([f"{k}: {v}" for k, v in entity.properties.items()])
            context_items.append(f"- {entity.type}: {entity.id} - {props_str}")
        
        for rel in context.relationships:
            context_items.append(f"- Relationship: {rel.source} -> {rel.type} -> {rel.target}")
        
        context_info = "\n".join(context_items) or "No context information available"
        
        # Format current plan
        current_plan_text = f"Goal: {plan.goal}\n\nPhases:\n"
        for i, phase in enumerate(plan.phases):
            current_plan_text += f"{i+1}. {phase.name} - {phase.description}\n"
            current_plan_text += f"   Success Criteria: {phase.success_criteria}\n"
        
        # Format phase progress information
        completed_phases_text = ""
        for i in range(plan.current_phase):
            completed_phases_text += f"{i+1}. {plan.phases[i].name}\n"
        
        if not completed_phases_text:
            completed_phases_text = "None yet"
        
        current_phase_text = f"Phase {plan.current_phase + 1}: {plan.phases[plan.current_phase].name}"
        
        remaining_phases_text = ""
        for i in range(plan.current_phase + 1, len(plan.phases)):
            remaining_phases_text += f"{i+1}. {plan.phases[i].name}\n"
        
        if not remaining_phases_text:
            remaining_phases_text = "None - this is the final phase"
        
        # Format observations
        observations_text = "\n".join([
            f"- {obs.get('tool', 'Unknown')}: {obs.get('result', 'No result')}"
            for obs in new_observations
        ]) or "No new observations"
        
        # Create the prompt
        prompt = self.plan_revision_template.format(
            current_plan=current_plan_text,
            context_info=context_info,
            completed_phases=completed_phases_text,
            current_phase=current_phase_text,
            remaining_phases=remaining_phases_text,
            recent_observations=observations_text
        )
        
        # Get revised plan from LLM
        plan_schema = {
            "type": "object",
            "properties": {
                "phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "success_criteria": {"type": "string"}
                        }
                    }
                },
                "assumptions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "confidence": {"type": "number"}
                        }
                    }
                },
                "reasoning": {"type": "string"}
            }
        }
        
        logger.info("Generating revised plan using LLM")
        revised_data = self.llm_adapter.generate_json(prompt, plan_schema)
        
        # Log reasoning
        reasoning = revised_data.get("reasoning", "No reasoning provided")
        logger.info(f"Plan revision reasoning: {reasoning}")
        
        # Convert JSON to plan objects
        revised_phases = []
        
        # Keep all completed phases from the original plan
        for i in range(plan.current_phase):
            revised_phases.append(plan.phases[i])
        
        # Add current and future phases from the revised plan
        for phase_data in revised_data.get("phases", []):
            phase = Phase(
                name=phase_data["name"],
                description=phase_data["description"],
                actions=[],  # Start with empty actions list
                success_criteria=phase_data["success_criteria"]
            )
            revised_phases.append(phase)
        
        assumptions = []
        for assumption_data in revised_data.get("assumptions", []):
            assumption = Assumption(
                description=assumption_data["description"],
                confidence=assumption_data.get("confidence", 1.0)
            )
            assumptions.append(assumption)
        
        # Create revised plan
        revised_plan = Plan(goal=plan.goal, phases=revised_phases, assumptions=assumptions)
        revised_plan.current_phase = plan.current_phase
        
        logger.info(f"Revised plan with {len(revised_phases)} phases and {len(assumptions)} assumptions")
        
        return revised_plan 