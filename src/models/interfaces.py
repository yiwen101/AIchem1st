from typing import Dict, List, Any, Optional
from enum import Enum


class ActionStatus(Enum):
    """Status enum for actions."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Tool:
    """Representation of a tool that can be used by the agent."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], 
                 function: callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        
    def __repr__(self) -> str:
        return f"Tool(name={self.name})"


class FallbackOption:
    """Fallback option for actions that fail."""
    
    def __init__(self, tool: str, params: Dict[str, Any], priority: int = 0):
        self.tool = tool
        self.params = params
        self.priority = priority


class Action:
    """Representation of a single action in a plan."""
    
    def __init__(self, tool: str, params: Dict[str, Any], purpose: str, 
                 fallbacks: List[FallbackOption] = None):
        self.tool = tool
        self.params = params
        self.purpose = purpose
        self.fallbacks = fallbacks or []
        self.status = ActionStatus.PENDING
        self.result = None
        self.error = None
        
    def __repr__(self) -> str:
        return f"Action(tool={self.tool}, status={self.status.value}), params={self.params}, purpose={self.purpose}, fallbacks={self.fallbacks}"


class Assumption:
    """Representation of an assumption made during planning."""
    
    def __init__(self, description: str, confidence: float = 1.0):
        self.description = description
        self.confidence = confidence
        self.validated = False
    
    def __repr__(self) -> str:
        return f"Assumption(description={self.description}, confidence={self.confidence})"


class Phase:
    """Representation of a phase in a plan."""
    
    def __init__(self, name: str, description: str, actions: List[Action], 
                 success_criteria: str):
        self.name = name
        self.description = description
        self.actions = actions
        self.success_criteria = success_criteria
        self.completed = False
        
    def __repr__(self) -> str:
        return f"Phase(name={self.name}, completed={self.completed}), actions={self.actions}"


class Plan:
    """Representation of a complete execution plan."""
    
    def __init__(self, goal: str, phases: List[Phase], assumptions: List[Assumption] = None):
        self.goal = goal
        self.phases = phases
        self.assumptions = assumptions or []
        self.current_phase = 0
        
    def __repr__(self) -> str:
        return f"Plan(goal={self.goal}\nphases={self.phases}\nassumptions={self.assumptions})"


class Entity:
    """Representation of an entity in the knowledge graph."""
    
    def __init__(self, id: str, type: str, properties: Dict[str, Any] = None):
        self.id = id
        self.type = type
        self.properties = properties or {}
        self.relevance_score = 1.0  # Initial relevance score
        
    def __repr__(self) -> str:
        return f"Entity(id={self.id}, type={self.type})"


class Relationship:
    """Representation of a relationship in the knowledge graph."""
    
    def __init__(self, source: str, target: str, type: str, 
                 properties: Dict[str, Any] = None):
        self.source = source
        self.target = target
        self.type = type
        self.properties = properties or {}
        self.relevance_score = 1.0  # Initial relevance score
        
    def __repr__(self) -> str:
        return f"Relationship({self.source} -> {self.type} -> {self.target})"


class Hypothesis:
    """Representation of a hypothesis in the knowledge graph."""
    
    def __init__(self, description: str, confidence: float = 0.5, 
                 supporting_evidence: List[str] = None):
        self.description = description
        self.confidence = confidence
        self.supporting_evidence = supporting_evidence or []
        
    def __repr__(self) -> str:
        return f"Hypothesis({self.description}, confidence={self.confidence})"


class ContextNote:
    """Container for context information."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.hypotheses: List[Hypothesis] = []
        
    def __repr__(self) -> str:
        return f"ContextNote(entities={len(self.entities)}, relationships={len(self.relationships)})" 