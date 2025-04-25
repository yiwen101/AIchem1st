#!/usr/bin/env python3
"""
Script to extract TQAgent plans for randomly selected questions in the development set.
The plans now include named parameters that can be referenced in subsequent steps.
"""

import json
import os
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.TQAgent import TQAgent
from app.model.structs import ParquetFileRow
from app.common.monitor import logger

def load_development_set() -> list[ParquetFileRow]:
    """Load questions from development_set.json"""
    with open("development_set.json", "r") as f:
        development_set = json.load(f)
        return [ParquetFileRow(**item) for item in development_set]

def main():
    """Extract TQAgent plans for randomly selected questions in development set."""
    print("Initializing TQAgent...")
    agent = TQAgent(num_frames=15, model="gpt-4o-mini")
    
    print("Loading development set...")
    rows = load_development_set()
    
    print(f"Loaded {len(rows)} questions from development set")
    
    # Create directory for plans if it doesn't exist
    plans_dir = "tq_agent_plans"
    if not os.path.exists(plans_dir):
        os.makedirs(plans_dir)
    
    # Randomly select 5 questions from the development set
    random.seed(42)  # For reproducibility
    selected_rows = random.sample(rows, 5)
    
    print(f"Randomly selected 5 questions for analysis")
    
    # Process each selected question
    all_plans = {}
    
    for i, row in enumerate(selected_rows):
        print(f"\nProcessing question {i+1}/5 (QID: {row.qid})")
        print(f"Question: {row.question}")
        print(f"Capability: {row.capability}")
        
        try:
            # Get the method sequence plan from TQAgent
            method_sequence = agent._determine_method_sequence(row.question)
            
            # Format the plan for better readability
            formatted_plan = []
            for step_num, step in enumerate(method_sequence, 1):
                method_name = step["method"]
                params = step["params"]
                result_name = step.get("result_name", "unnamed_result")
                
                formatted_step = {
                    "step": step_num,
                    "method": method_name,
                    "parameters": params,
                    "result_name": result_name
                }
                
                formatted_plan.append(formatted_step)
            
            # Store the plan with the question details
            plan_data = {
                "qid": row.qid,
                "question": row.question,
                "capability": row.capability,
                "video_id": row.video_id,
                "plan": formatted_plan
            }
            
            # Add to all plans
            all_plans[row.qid] = plan_data
            
            # Save individual plan to file
            individual_plan_path = os.path.join(plans_dir, f"plan_{row.qid}.json")
            with open(individual_plan_path, 'w') as f:
                json.dump(plan_data, f, indent=2)
            
            print(f"Generated plan with {len(formatted_plan)} steps")
            print(f"Plan saved to: {individual_plan_path}")
            
            # Print a brief summary of the plan
            print("\nPlan Summary:")
            for step in formatted_plan:
                # Check for parameter values starting with $
                param_refs = []
                for param_name, param_value in step["parameters"].items():
                    if isinstance(param_value, str) and param_value.startswith('$'):
                        param_refs.append(f"{param_name}={param_value}")
                
                param_refs_str = f" [{', '.join(param_refs)}]" if param_refs else ""
                print(f"  Step {step['step']}: {step['method']} → {step['result_name']}{param_refs_str}")
            
        except Exception as e:
            print(f"Error creating plan for question {row.qid}: {str(e)}")
            logger.log_exception(e, f"Error creating plan for question {row.qid}")
    
    # Save all plans to a single file
    all_plans_path = os.path.join(plans_dir, "all_tq_agent_plans.json")
    with open(all_plans_path, 'w') as f:
        json.dump(all_plans, f, indent=2)
    
    print(f"\nAll plans saved to: {all_plans_path}")
    print(f"Total plans created: {len(all_plans)}/5")

if __name__ == "__main__":
    main()
