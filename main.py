#!/usr/bin/env python3
"""
Run script for the QAAgent.

This script loads the QAAgent and runs it on the evaluation dataset.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#from app.QAAgent import QAAgent
from app.NaiveAgent import NaiveAgent
from app.TQAgent import TQAgent
from app.L3Agent import L3Agent
from app.hypothesis_based_agent import HypothesisBasedAgent
from eval import evaluate_video_agent_on_mcq_part, generate_development_set_result

def main():
    """Main function to run the QAAgent against evaluation datasets."""
    print("Initializing QAAgent...")
    #agent = QAAgent(max_steps=20)
    #agent = NaiveAgent(num_frames=10, require_explanation=True)
    #agent = TQAgent(num_frames=10, display=True)
    #agent = L3Agent(display=True)
    agent = HypothesisBasedAgent(display=False)
    '''
    # Evaluate on MCQ parts
    print("\nEvaluating on MCQ parts...")
    mcq_part_indexes = [5]  # Can be adjusted to include more parts
    evaluate_video_agent_on_mcq_part(agent, mcq_part_indexes)
    
    # Generate development set results
    print("\nGenerating development set results...")
    '''
    generate_development_set_result(agent)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 