import pandas as pd
import os

from app.model.structs import ParquetFileRow

all_question_file_name = "test-00000-of-00001.parquet"

df = pd.read_parquet(all_question_file_name)
#{"qid":"0008-2","video_id":"sj81PWrerDk","question_type":"Correctly-led Open-ended Question","capability":"Plot Attribute (Montage)","question":"Did the last person open the bottle without using a knife?","duration":"8.85","question_prompt":"Please state your answer with a brief explanation.","answer":"","youtube_url":"https://www.youtube.com/shorts/sj81PWrerDk"}
# qid, video_id, question_type, capability, question, duration, question_prompt, answer, youtube_url
print(df.head())

# to a list of ParquetFileRow
parquet_file_rows = [ParquetFileRow(**row) for index, row in df.iterrows()]

print(parquet_file_rows)

