import pandas as pd
import os
import atexit

from app.model.structs import ParquetFileRow
from app.graph import create_video_agent_graph
from app.tools.resource.resource_manager import resource_manager
from app.common.monitor import logger
from langchain_core.runnables import RunnableConfig

def preload_video(video_id):
    """Preload a video into the resource manager."""
    video_path = f"videos/{video_id}.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        logger.log_warning(f"Video file not found: {video_path}")
        return False
    
    try:
        # Load video into resource manager
        metadata = resource_manager.load_video(video_path)
        logger.log_info(f"Loaded video {video_id} - Duration: {metadata['duration']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")
        return True
    except Exception as e:
        logger.log_error(f"Error loading video {video_id}: {str(e)}")
        return False

def cleanup_resources():
    """Clean up resources on exit."""
    logger.log_info("Cleaning up resources...")
    resource_manager.cleanup()
    logger.log_info("Resource cleanup complete")

# Register cleanup function to run on exit
atexit.register(cleanup_resources)

# Create videos directory if it doesn't exist
os.makedirs("videos", exist_ok=True)

# Create output directories
os.makedirs("app/tools/output/image_captioning", exist_ok=True)
os.makedirs("app/tools/output/object_detection", exist_ok=True)
os.makedirs("app/tools/output/scene_detection", exist_ok=True)

# Main entry point
logger.log_info("Initializing AIchem1st video understanding agent")
logger.log_info("Loading dataset...")
all_question_file_name = "test-00000-of-00001.parquet"

df = pd.read_parquet(all_question_file_name)
#{"qid":"0008-2","video_id":"sj81PWrerDk","question_type":"Correctly-led Open-ended Question","capability":"Plot Attribute (Montage)","question":"Did the last person open the bottle without using a knife?","duration":"8.85","question_prompt":"Please state your answer with a brief explanation.","answer":"","youtube_url":"https://www.youtube.com/shorts/sj81PWrerDk"}
# qid, video_id, question_type, capability, question, duration, question_prompt, answer, youtube_url
print(df.head())

# Convert to a list of ParquetFileRow
parquet_file_rows = [ParquetFileRow(**row) for index, row in df.iterrows()]

# Create the graph with max_steps=20
logger.log_info("Creating agent graph...")
graph = create_video_agent_graph(max_steps=20)
logger.log_info("Agent graph created successfully")

# Process queries one by one
logger.log_info("Starting to process queries...")
count = 0
for i in range(len(parquet_file_rows)):
    if i % 5 != 0:
        continue
    row = parquet_file_rows[i]
    
    # Load the video for this query
    logger.log_info(f"Loading video for query {row.qid}: {row.video_id}")
    if not preload_video(row.video_id):
        logger.log_error(f"Could not load video for {row.qid}, skipping...")
        continue
    
    logger.log_info(f"Processing query {i+1}/{len(parquet_file_rows)}: {row.qid}")
    graph.invoke(input={
        "query": row,

        "qa_notebook": [],
        "tool_results": {},

        "question_stack": [row.question],
        "task_queue": [],

        "current_question_tool_results": {},
        "previous_QA": None,
        "prev_attempt_answer_response": None,
        
        # Initialize step count tracking
        "step_count": 0,
        "max_steps": 20
    }, config={
    "recursion_limit": 100  # Increase the limit as needed
    })
    logger.log_info(f"Finished processing query {row.qid}")
    count += 1
    if count > 5:
        break

logger.log_info("All queries processed successfully")

# Resource cleanup happens automatically via atexit handler

'''
options = ["A", "B", "C", "D", "E"]
for option in options:
    with open(f"result_{option}.csv", "w") as f:
        f.write("qid,pred\n")

    for i in range(len(parquet_file_rows)):
        row = parquet_file_rows[i]
        with open(f"result_{option}.csv", "a") as f:
            f.write(f"{row.qid},{option}\n")
            '''