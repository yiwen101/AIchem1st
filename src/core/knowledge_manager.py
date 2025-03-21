from typing import Dict, List, Optional, Any
import copy

from src.models.interfaces import Entity, Relationship, Hypothesis, ContextNote
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


class KnowledgeManager:
    """
    Manages the knowledge graph context for the agent.
    """
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.context = ContextNote()
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity: The entity to add
        """
        if entity.id in self.context.entities:
            logger.info(f"Updating existing entity: {entity.id}")
            # Merge properties of existing entity with new entity
            for key, value in entity.properties.items():
                self.context.entities[entity.id].properties[key] = value
        else:
            logger.info(f"Adding new entity: {entity.id} ({entity.type})")
            self.context.entities[entity.id] = entity
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the knowledge graph.
        
        Args:
            relationship: The relationship to add
        """
        # Check if source and target entities exist
        if relationship.source not in self.context.entities:
            logger.warning(f"Source entity '{relationship.source}' does not exist for relationship")
            return
        
        if relationship.target not in self.context.entities:
            logger.warning(f"Target entity '{relationship.target}' does not exist for relationship")
            return
        
        # Check if this relationship already exists
        for existing_rel in self.context.relationships:
            if (existing_rel.source == relationship.source and
                existing_rel.target == relationship.target and
                existing_rel.type == relationship.type):
                logger.info(f"Updating existing relationship: {relationship.source} -> {relationship.type} -> {relationship.target}")
                # Merge properties
                for key, value in relationship.properties.items():
                    existing_rel.properties[key] = value
                return
        
        # Add new relationship
        logger.info(f"Adding new relationship: {relationship.source} -> {relationship.type} -> {relationship.target}")
        self.context.relationships.append(relationship)
    
    def add_relationship_if_entities_exist(self, source: str, target: str, rel_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add a relationship only if both source and target entities exist.
        
        Args:
            source: Source entity ID
            target: Target entity ID
            rel_type: Relationship type
            properties: Optional properties for the relationship
            
        Returns:
            True if relationship was added, False otherwise
        """
        if source not in self.context.entities:
            logger.warning(f"Source entity '{source}' does not exist for relationship")
            return False
        
        if target not in self.context.entities:
            logger.warning(f"Target entity '{target}' does not exist for relationship")
            return False
            
        # Create and add the relationship
        relationship = Relationship(
            source=source,
            target=target,
            type=rel_type,
            properties=properties or {}
        )
        
        # Check if this relationship already exists
        for existing_rel in self.context.relationships:
            if (existing_rel.source == source and
                existing_rel.target == target and
                existing_rel.type == rel_type):
                logger.info(f"Updating existing relationship: {source} -> {rel_type} -> {target}")
                # Merge properties
                for key, value in relationship.properties.items():
                    existing_rel.properties[key] = value
                return True
        
        # Add new relationship
        logger.info(f"Adding new relationship: {source} -> {rel_type} -> {target}")
        self.context.relationships.append(relationship)
        return True
    
    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """
        Add a hypothesis to the knowledge graph.
        
        Args:
            hypothesis: The hypothesis to add
        """
        # Check if this hypothesis already exists
        for existing_hyp in self.context.hypotheses:
            if existing_hyp.description == hypothesis.description:
                logger.info(f"Updating existing hypothesis: {hypothesis.description}")
                existing_hyp.confidence = hypothesis.confidence
                existing_hyp.supporting_evidence = list(set(existing_hyp.supporting_evidence + hypothesis.supporting_evidence))
                return
        
        # Add new hypothesis
        logger.info(f"Adding new hypothesis: {hypothesis.description}")
        self.context.hypotheses.append(hypothesis)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The ID of the entity to retrieve
            
        Returns:
            The entity if found, None otherwise
        """
        return self.context.entities.get(entity_id)
    
    def get_relationships(self, source: Optional[str] = None, target: Optional[str] = None, 
                          rel_type: Optional[str] = None) -> List[Relationship]:
        """
        Get relationships matching the given criteria.
        
        Args:
            source: Source entity ID (optional)
            target: Target entity ID (optional)
            rel_type: Relationship type (optional)
            
        Returns:
            List of matching relationships
        """
        results = []
        
        for rel in self.context.relationships:
            if (source is None or rel.source == source) and \
               (target is None or rel.target == target) and \
               (rel_type is None or rel.type == rel_type):
                results.append(rel)
        
        return results
    
    def get_context_snapshot(self, max_entities: int = 30, 
                             max_relationships: int = 50,
                             max_hypotheses: int = 10) -> ContextNote:
        """
        Get a snapshot of the current context, limited to the most relevant items.
        
        Args:
            max_entities: Maximum number of entities to include
            max_relationships: Maximum number of relationships to include
            max_hypotheses: Maximum number of hypotheses to include
            
        Returns:
            A context snapshot
        """
        snapshot = ContextNote()
        
        # Sort entities by relevance score
        sorted_entities = sorted(
            self.context.entities.values(),
            key=lambda e: e.relevance_score,
            reverse=True
        )
        
        # Add top entities to snapshot
        entity_ids = set()
        for entity in sorted_entities[:max_entities]:
            snapshot.entities[entity.id] = copy.deepcopy(entity)
            entity_ids.add(entity.id)
        
        # Sort relationships by relevance score
        sorted_relationships = sorted(
            self.context.relationships,
            key=lambda r: r.relevance_score,
            reverse=True
        )
        
        # Add top relationships to snapshot, but only if both entities are included
        for rel in sorted_relationships:
            if len(snapshot.relationships) >= max_relationships:
                break
                
            if rel.source in entity_ids and rel.target in entity_ids:
                snapshot.relationships.append(copy.deepcopy(rel))
        
        # Sort hypotheses by confidence
        sorted_hypotheses = sorted(
            self.context.hypotheses,
            key=lambda h: h.confidence,
            reverse=True
        )
        
        # Add top hypotheses to snapshot
        for hyp in sorted_hypotheses[:max_hypotheses]:
            snapshot.hypotheses.append(copy.deepcopy(hyp))
        
        return snapshot
    
    def clear(self) -> None:
        """Clear all knowledge from the context."""
        self.context = ContextNote()
        logger.info("Knowledge context cleared")
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update the knowledge graph from a dictionary representation.
        
        Args:
            data: Dictionary containing entities, relationships, and hypotheses
        """
        # Add entities
        for entity_data in data.get("entities", []):
            entity = Entity(
                id=entity_data["id"],
                type=entity_data["type"],
                properties=entity_data.get("properties", {})
            )
            self.add_entity(entity)
        
        # Add relationships
        for rel_data in data.get("relationships", []):
            relationship = Relationship(
                source=rel_data["source"],
                target=rel_data["target"],
                type=rel_data["type"],
                properties=rel_data.get("properties", {})
            )
            self.add_relationship(relationship)
        
        # Add hypotheses
        for hyp_data in data.get("hypotheses", []):
            hypothesis = Hypothesis(
                description=hyp_data["description"],
                confidence=hyp_data.get("confidence", 0.5),
                supporting_evidence=hyp_data.get("supporting_evidence", [])
            )
            self.add_hypothesis(hypothesis) 