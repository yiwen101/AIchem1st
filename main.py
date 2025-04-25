from typing import Annotated

import typer

cli = typer.Typer()

@cli.command()
def run_l3(
    file: Annotated[
        str, typer.Argument(help="The parquet file to run the L3 agent on")
    ],
    output_dir: Annotated[str, typer.Option(help="The output directory")] = "results",
    start_index: Annotated[int, typer.Option(help="The start index")] = 0,
    end_index: Annotated[int, typer.Option(help="The end index")] = -1,
):
    """Run the L3 agent"""

    from app.L3Agent import L3Agent

    agent = L3Agent(display=False)

    questions, start_index, end_index = __process_questions(
        file, start_index, end_index
    )

    __run_video_agent_all(agent, questions, output_dir, start_index, end_index)


@cli.command()
def run_l3_single(
    file: Annotated[
        str, typer.Argument(help="The parquet file to run the L3 agent on")
    ],
    question_id: Annotated[str, typer.Argument(help="The question id")],
):
    """Run the L3 agent on a single question"""
    from app.L3Agent import L3Agent

    agent = L3Agent(display=False)

    questions, _, _ = __process_questions(file)
    questions = [qns for qns in questions if qns.qid == question_id]
    if len(questions) == 0:
        raise ValueError(f"Question {question_id} not found in {file}")

    __run_video_agent_single(agent, questions[0])


def __process_questions(
    questions_parquet_file: str,
    start_index: int = 0,
    end_index: int = -1,
):
    """Process the questions in the parquet file."""
    from pathlib import Path

    from app.common import load_questions_parquet

    # Load questions
    questions = load_questions_parquet(Path(questions_parquet_file))

    # Set start index
    if start_index < 0:
        start_index = max(len(questions) + start_index, 0)
    else:
        start_index = min(start_index, len(questions))

    # Set end index
    if end_index < 0:
        end_index = max(len(questions) + end_index, 0) + 1
    else:
        end_index = min(end_index + 1, len(questions))

    questions = questions[start_index:end_index]
    return questions, start_index, end_index


def __run_video_agent_single(
    agent,  # IVideoAgent
    question,  # ParquetFileRow
):
    """Run the video agent on the question."""
    answer = agent.get_cleaned_answer(question)
    print(answer)


def __run_video_agent_all(
    agent,  # IVideoAgent
    questions,  # list[ParquetFileRow]
    output_dir: str,
    start_index: int = 0,
    end_index: int = -1,
):
    """Run the video agent on the questions in the parquet file."""
    import logging
    from pathlib import Path

    from app.common import logger

    logger.logger.setLevel(logging.INFO)

    # Create output directory
    eval_result_dir = Path(output_dir)
    if not eval_result_dir.exists():
        eval_result_dir.mkdir(parents=True)

    eval_result_file = (
        eval_result_dir
        / f"{agent.get_agent_name()}_{start_index + 1:04d}_{end_index + 1:04d}.csv"
    )
    print(f"Processing {len(questions)} questions, from {start_index} to {end_index}")

    # Create eval result file
    if not eval_result_file.exists():
        with open(eval_result_file, "w") as f:
            f.write("qid,pred\n")

    # Generate answers
    with open(eval_result_file, "a") as f:
        # TODO: Use asyncio to generate answers
        for row in questions:
            logger.log_info(
                f"Generating answer for {agent.get_agent_name()} on question {row.qid}"
            )
            answer = agent.get_cleaned_answer(row)
            f.write(f"{row.qid},{answer}\n")
            f.flush()

    print(f"Done! Answers saved to {eval_result_file}")


if __name__ == "__main__":
    cli()
