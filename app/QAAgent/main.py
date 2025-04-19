"""
Test script for the QAAgent.

This script loads the QAAgent and runs it on test data.
"""

import pandas as pd
import os

from app.model.structs import ParquetFileRow
from app.QAAgent import QAAgent
from app.common.monitor import logger

def main():
    """Main entry point for testing the QAAgent."""
    logger.log_info("Initializing AIchem1st video understanding agent")
    
    # Load the dataset
    logger.log_info("Loading dataset...")
    all_question_file_name = "test-00000-of-00001.parquet"
    
    if not os.path.exists(all_question_file_name):
        logger.log_error(f"Dataset file not found: {all_question_file_name}")
        return
    
    df = pd.read_parquet(all_question_file_name)
    logger.log_info(f"Loaded {len(df)} questions from dataset")
    print(df.head())
    
    # Convert to a list of ParquetFileRow
    parquet_file_rows = [ParquetFileRow(**row) for index, row in df.iterrows()]
    
    # Create the agent
    logger.log_info("Creating QAAgent...")
    agent = QAAgent(max_steps=20)
    logger.log_info("QAAgent created successfully")
    
    # Process queries one by one
    logger.log_info("Starting to process queries...")
    count = 0
    for i in range(len(parquet_file_rows)):
        if i % 5 != 0:  # Process every 5th query for testing
            continue
        
        row = parquet_file_rows[i]
        logger.log_info(f"Processing query {i+1}/{len(parquet_file_rows)}: {row.qid}")
        
        answer = agent.get_answer(row)
        
        logger.log_info(f"Query: {row.question}")
        logger.log_info(f"Answer: {answer}")
        logger.log_info("-" * 80)
        
        count += 1
        if count > 5:  # Process at most 5 queries for testing
            break
    
    logger.log_info("All queries processed successfully")

if __name__ == "__main__":
    main()
