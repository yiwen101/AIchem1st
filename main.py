import pandas as pd
import os

from app.model.structs import ParquetFileRow
from app.graph import create_video_agent_graph

all_question_file_name = "test-00000-of-00001.parquet"

df = pd.read_parquet(all_question_file_name)
#{"qid":"0008-2","video_id":"sj81PWrerDk","question_type":"Correctly-led Open-ended Question","capability":"Plot Attribute (Montage)","question":"Did the last person open the bottle without using a knife?","duration":"8.85","question_prompt":"Please state your answer with a brief explanation.","answer":"","youtube_url":"https://www.youtube.com/shorts/sj81PWrerDk"}
# qid, video_id, question_type, capability, question, duration, question_prompt, answer, youtube_url
print(df.head())

# to a list of ParquetFileRow
parquet_file_rows = [ParquetFileRow(**row) for index, row in df.iterrows()]

graph = create_video_agent_graph()

for i in range(len(parquet_file_rows)):
    row = parquet_file_rows[i]
    graph.invoke(input={
        "query": row,

        "qa_notebook": [],
        "tool_results": {},

        "question_stack": [row.question],
        "task_queue": [],

        "current_question_tool_results": {},
        "previous_QA": None,
        "prev_attempt_answer_response": None
    })

