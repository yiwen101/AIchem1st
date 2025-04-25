import re
from app.common.questions_loader import ParquetFileRow
from eval import load_all_questions
import os

def recover_answer_from_log(log_dir, add_explanation=False):
    rows = load_all_questions()
    mcq_questions = [row for row in rows if row.is_mcq()]
    mcq_question_id = {}
    for row in mcq_questions:
        mcq_question_id[row.qid] = True

    all_answer = []
    all_explanation = []
    all_question_id = []

    # itereate throu all files under process folder

    for file in os.listdir(log_dir):
        with open(f"{log_dir}/{file}", "r") as f:
            last_question_id = None
            last_answer = None
            last_explanation = None
            for line in f:
                if 'answer=' in line and 'explanation=' in line:
                    parts = line.split('explanation=')
                    explanation = parts[1].strip()
                    explanation = explanation.split('need_zoom_in_to_part_of_video=')[0].strip()
                    explanation = explanation.split('description=')[0].strip()
                    answer = parts[0].split('answer=')[1].strip()
                    last_answer = answer
                    last_explanation = explanation
                else:
                    if 'Generating answer for L3Agent_gpt-4o-mini on question ' in line:
                        question_id = line.split('Generating answer for L3Agent_gpt-4o-mini on question ')[1].strip()
                        if last_question_id == None:
                            last_question_id = question_id
                        else:
                            if last_question_id not in all_question_id:
                                all_question_id.append(last_question_id)
                                last_question_id = question_id
                                all_answer.append(last_answer)
                                all_explanation.append(last_explanation)
                                last_answer = None
                                last_explanation = None
                            else:
                                last_question_id = question_id
                                last_answer = None
                                last_explanation = None
            if last_question_id not in all_question_id:
                all_question_id.append(last_question_id)
                all_answer.append(last_answer)
                all_explanation.append(last_explanation)
        print(f"Processed {file}, current number of questions: {len(all_question_id)}")


    with open("result_from_log.csv", "w") as f:
        f.write("qid,pred\n")
        for answer, explanation, question_id in zip(all_answer, all_explanation, all_question_id):
            if question_id in mcq_question_id:
                f.write(f"{question_id},{answer}\n")
            else:
                if add_explanation:
                    f.write(f"{question_id},{answer[:-2]}. {explanation[1:]}\n")
                else:
                    f.write(f"{question_id},{answer}\n")

if __name__ == "__main__":
    recover_answer_from_log("submission/final_submission_logs", add_explanation=False)