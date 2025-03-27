#!/usr/bin/env python
"""
Run script for the video agent.

This script provides a simple way to run the video agent from the command line.
"""

import os
import sys
import argparse

# Add current directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import dotenv to load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from app.main import main as app_main
from examples.simple_example import main as simple_example_main
from examples.graph_visualization import main as graph_viz_main
from examples.print_graph import main as print_graph_main


def main():
    """Run the specified example or the main application."""
    parser = argparse.ArgumentParser(description="Video Agent Runner")
    
    parser.add_argument(
        "--example", "-e",
        choices=["simple", "graph_viz", "print_graph"],
        help="Run a specific example (simple, graph_viz, print_graph)"
    )
    
    # Parse the example argument only
    args, remaining_args = parser.parse_known_args()
    
    if args.example:
        # Run the specified example
        print(f"Running example: {args.example}")
        
        if args.example == "simple":
            simple_example_main()
        elif args.example == "graph_viz":
            graph_viz_main()
        elif args.example == "print_graph":
            print_graph_main()
    else:
        # Run the main application
        sys.argv = [sys.argv[0]] + remaining_args
        app_main()


if __name__ == "__main__":
    main() 