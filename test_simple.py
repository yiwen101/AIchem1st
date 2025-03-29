"""
Simple test to verify the JSON serialization fixes.
"""

from app.model.structs import ParquetFileRow
from app.nodes.is_primitive_question import node_response_schema
from app.nodes.try_answer_with_reasoning import node_response_schema as reasoning_schema
import json

def main():
    """Test JSON serialization of schemas."""
    # Test is_primitive_question schema
    print("Testing is_primitive_question schema serialization:")
    schema_json = json.dumps(node_response_schema, indent=2)
    print(schema_json)
    
    # Test try_answer_with_reasoning schema
    print("\nTesting try_answer_with_reasoning schema serialization:")
    reasoning_json = json.dumps(reasoning_schema, indent=2)
    print(reasoning_json)
    
    print("\nJSON serialization successful!")

if __name__ == "__main__":
    main() 