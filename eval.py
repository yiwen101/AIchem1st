import json
import pandas as pd
import os
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

def load_development_set() -> list[ParquetFileRow]:
    with open("development_set.json", "r") as f:
        development_set = json.load(f)
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

def generate_development_set_result(video_agent: IVideoAgent):
    logger.log_info(f"Generating development set result for {video_agent.get_agent_name()}")
    rows = load_development_set()
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










