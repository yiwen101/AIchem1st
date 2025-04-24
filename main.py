#!/usr/bin/env python3
from app.L3Agent import L3Agent
from app.common.monitor import logger
from eval import generate_answer_for_all

def main():
    logger.log_info("Starting evaluation...")
    agent = L3Agent(display=False)
    generate_answer_for_all(agent)
    logger.log_info("Evaluation complete!")

if __name__ == "__main__":
    main() 