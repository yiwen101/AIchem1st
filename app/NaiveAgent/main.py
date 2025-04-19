"""
Test script for the NaiveAgent.

This script loads the NaiveAgent and runs it on test data.
"""

import pandas as pd
import os

from app.model.structs import ParquetFileRow
from app.NaiveAgent import NaiveAgent
from app.common.monitor import logger

def main():
    """Main entry point for testing the NaiveAgent."""
    logger.log_info("Initializing NaiveAgent")
    
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
    logger.log_info("Creating NaiveAgent...")
    agent = NaiveAgent(num_frames=10, model="gpt-4o-mini")
    logger.log_info("NaiveAgent created successfully")
    
    # Process queries one by one
    logger.log_info("Starting to process queries...")
    count = 0
    for i in range(len(parquet_file_rows)):
        if i % 10 != 0:  # Process every 10th query for testing (to limit API usage)
            continue
        
        row = parquet_file_rows[i]
        logger.log_info(f"Processing query {i+1}/{len(parquet_file_rows)}: {row.qid}")
        
        answer = agent.get_answer(row)
        
        logger.log_info(f"Query: {row.question}")
        logger.log_info(f"Answer: {answer}")
        logger.log_info("-" * 80)
        
        count += 1
        if count >= 3:  # Process at most 3 queries for testing (to limit API usage)
            break
    
    logger.log_info("All queries processed successfully")

if __name__ == "__main__":
    main() 