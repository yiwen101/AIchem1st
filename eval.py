import json
import pandas as pd
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Dict
from app.model.interface import IVideoAgent
from app.model.structs import ParquetFileRow
from app.common.monitor import logger


def load_mcq_part(index: int) -> list[ParquetFileRow]:
    file_path = f"mcq_parts/mcq_part_{index}"
    parquet_file_path = f"{file_path}.parquet"
    json_file_path = f"{file_path}.json"
    if not os.path.exists(parquet_file_path):
        raise FileNotFoundError(f"File {parquet_file_path} not found")
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File {json_file_path} not found")

    labels = {}
    with open(json_file_path, "r") as f:
        data = json.load(f)
        for item in data:
            labels[item["qid"]] = item["label"]

    df = pd.read_parquet(parquet_file_path)
    parquet_file_rows = [ParquetFileRow(**row) for index, row in df.iterrows()]
    for row in parquet_file_rows:
        if row.qid not in labels:
            raise ValueError(f"QID {row.qid} not found in labels")
        row.label = labels[row.qid]
    return parquet_file_rows

def load_development_set(shuffle: bool = True) -> list[ParquetFileRow]:
    with open("development_set.json", "r") as f:
        development_set = json.load(f)
        if shuffle:
            import random
            random.shuffle(development_set)
        return [ParquetFileRow(**item) for item in development_set]


def load_all_questions() -> list[ParquetFileRow]:
    df = pd.read_parquet("test-00000-of-00001.parquet")
    parquet_file_rows = [ParquetFileRow(**row) for index, row in df.iterrows()]
    return parquet_file_rows

def evaluate_video_agent_on_mcq_part(video_agent: IVideoAgent, mcq_part_indexes: list[int] = [5]):
    logger.log_info(f"Evaluating {video_agent.get_agent_name()} on MCQ parts {mcq_part_indexes}")
    rows = []
    for index in mcq_part_indexes:
        rows.extend(load_mcq_part(index))
    predicted_answers = []
    correct_answers = 0
    total_question_number = len(rows)
    for row in rows:
        logger.log_info(f"Evaluating {video_agent.get_agent_name()} on question {row.qid}")
        answer = video_agent.get_cleaned_answer(row)
        predicted_answers.append(answer)
        if answer == row.label:
            correct_answers += 1
    
    accuracy = correct_answers / total_question_number
    logger.log_info(f"Accuracy: {accuracy}")
    
    # create eval_result dir if not exists
    eval_result_dir = "eval_result"
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)
    
    agent_name = video_agent.get_agent_name()
    mcq_part_indexes_str = "_".join(str(index) for index in mcq_part_indexes)
    # create the dir if not exists
    eval_result_dir = f"{eval_result_dir}/mcq"
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)
    # write the result to a file
    with open(f"{eval_result_dir}/{agent_name}_{mcq_part_indexes_str}.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Total questions: {total_question_number}\n")
        f.write("-"*100 + "\n")
        f.write("Wrong answers:\n")
        for i in range(total_question_number):
            if predicted_answers[i] == rows[i].label:
                continue
            f.write(f"Question {i+1}: {rows[i].question} (Video ID: {rows[i].video_id}) (Correct answer: {rows[i].label}) (Predicted answer: {predicted_answers[i]})\n")

def generate_development_set_result(video_agent: IVideoAgent, shuffle: bool = True):
    logger.log_info(f"Generating development set result for {video_agent.get_agent_name()}")
    rows = load_development_set(shuffle=shuffle)
    eval_result_dir = f"eval_result/development_set"
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)
    result_file_name = f"{eval_result_dir}/{video_agent.get_agent_name()}.csv"
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    # create the dir if not exists
    with open(result_file_name, "w") as f:
        f.write("qid,pred\n")
        for row in rows:
            logger.log_info(f"Generating development set result for {video_agent.get_agent_name()} on question {row.qid}")
            answer = video_agent.get_cleaned_answer(row)
            f.write(f"{row.qid},{answer}\n")
            f.flush()

def generate_development_set_result_threaded(new_video_agent_func: Callable[[], IVideoAgent], num_threads: int = 2):
    """
    Generate development set results using multiple threads.
    
    Args:
        new_video_agent_func: A factory function that creates a new video agent instance for each thread
        num_threads: Number of threads to use (default: 5)
    """
    logger.log_info(f"Generating development set result using {num_threads} threads")
    
    # Load all rows from development set
    rows = load_development_set(shuffle=False)  # Don't shuffle to maintain order
    total_rows = len(rows)
    logger.log_info(f"Total questions in development set: {total_rows}")
    
    # Create output directory
    eval_result_dir = f"eval_result/development_set"
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)
    
    # Get agent name from a temporary instance
    temp_agent = new_video_agent_func()
    agent_name = temp_agent.get_agent_name()
    result_file_name = f"{eval_result_dir}/{agent_name}_threaded.csv"
    
    # Initialize file with header
    with open(result_file_name, "w") as f:
        f.write("qid,pred\n")
    
    # Create thread-safe structures
    results_lock = threading.Lock()
    task_queue = queue.Queue()
    completed_count = [0]  # Use a list to make it mutable in nested functions
    completed_lock = threading.Lock()
    
    # Add all tasks to queue
    for row in rows:
        task_queue.put(row)
    
    def worker():
        # Each thread creates its own agent
        video_agent = new_video_agent_func()
        worker_name = threading.current_thread().name
        logger.log_info(f"Worker {worker_name} started with agent {video_agent.get_agent_name()}")
        
        while True:
            try:
                # Get next row from queue (non-blocking)
                try:
                    row = task_queue.get(block=False)
                except queue.Empty:
                    # No more tasks
                    break
                
                # Process the row
                try:
                    logger.log_info(f"Worker {worker_name} processing question {row.qid}")
                    answer = video_agent.get_cleaned_answer(row)
                    
                    # Write result to file (thread-safe)
                    with results_lock:
                        with open(result_file_name, "a") as f:
                            f.write(f"{row.qid},{answer}\n")
                    
                    # Update progress counter
                    with completed_lock:
                        completed_count[0] += 1
                        logger.log_info(f"Progress: {completed_count[0]}/{total_rows} ({(completed_count[0]/total_rows)*100:.1f}%)")
                
                except Exception as e:
                    logger.log_error(f"Error processing row {row.qid}: {str(e)}")
                    # Write error result
                    with results_lock:
                        with open(result_file_name, "a") as f:
                            f.write(f"{row.qid},ERROR\n")
                
                finally:
                    # Mark task as done
                    task_queue.task_done()
            
            except Exception as e:
                logger.log_error(f"Worker {worker_name} encountered error: {str(e)}")
        
        # Clean up resources
        try:
            video_agent._cleanup_resources()
        except:
            pass
        logger.log_info(f"Worker {worker_name} finished")
    
    # Start worker threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, name=f"Thread-{i+1}")
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    logger.log_info(f"All threads completed. Results saved to {result_file_name}")

# Alternative implementation using ThreadPoolExecutor
def generate_development_set_result_pool(new_video_agent_func: Callable[[], IVideoAgent], num_threads: int = 5):
    """
    Generate development set results using ThreadPoolExecutor.
    
    Args:
        new_video_agent_func: A factory function that creates a new video agent instance for each thread
        num_threads: Number of threads to use (default: 5)
    """
    logger.log_info(f"Generating development set result using ThreadPoolExecutor with {num_threads} threads")
    
    # Load all rows from development set
    rows = load_development_set(shuffle=False)
    total_rows = len(rows)
    
    # Create output directory
    eval_result_dir = f"eval_result/development_set"
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)
    
    # Get agent name from a temporary instance
    temp_agent = new_video_agent_func()
    agent_name = temp_agent.get_agent_name()
    result_file_name = f"{eval_result_dir}/{agent_name}_pool.csv"
    
    # Initialize file with header
    with open(result_file_name, "w") as f:
        f.write("qid,pred\n")
    
    # Thread-local storage for agents
    thread_local = threading.local()
    results_lock = threading.Lock()
    processed_count = [0]
    count_lock = threading.Lock()
    
    def get_agent():
        if not hasattr(thread_local, "agent"):
            thread_local.agent = new_video_agent_func()
        return thread_local.agent
    
    def process_row(row):
        try:
            agent = get_agent()
            logger.log_info(f"Processing question {row.qid}")
            
            answer = agent.get_cleaned_answer(row)
            
            with results_lock:
                with open(result_file_name, "a") as f:
                    f.write(f"{row.qid},{answer}\n")
            
            with count_lock:
                processed_count[0] += 1
                logger.log_info(f"Progress: {processed_count[0]}/{total_rows} ({(processed_count[0]/total_rows)*100:.1f}%)")
            
            return row.qid, answer
        
        except Exception as e:
            logger.log_error(f"Error processing row {row.qid}: {str(e)}")
            with results_lock:
                with open(result_file_name, "a") as f:
                    f.write(f"{row.qid},ERROR\n")
            return row.qid, "ERROR"
    
    # Process rows with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(process_row, row) for row in rows]
        
        # Wait for all futures to complete (alternative to using as_completed)
        for future in futures:
            future.result()  # This will raise any exceptions that occurred
    
    # Clean up resources for all agents
    logger.log_info("All processing complete, cleaning up resources...")
    
    logger.log_info(f"Results saved to {result_file_name}")










