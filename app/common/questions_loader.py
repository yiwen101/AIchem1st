from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd


class QuestionType(Enum):
    MCQ = "Multiple-choice Question with a Single Correct Answer"
    CORRECTLY_LED = "Correctly-led Open-ended Question"
    PRIMARY_OPEN_ENDED = "Primary Open-ended Question"
    PARAPHRASED_OPEN_ENDED = "Paraphrased Open-ended Question"
    WRONGLY_LED = "Wrongly-led Open-ended Question"


@dataclass
class ParquetFileRow:
    qid: str
    video_id: str
    question_type: str
    capability: str
    question: str
    duration: str
    question_prompt: str
    answer: str
    youtube_url: str

    def is_mcq(self):
        return self.question_type == QuestionType.MCQ.value

    def is_correctly_led_question(self):
        return self.question_type == QuestionType.CORRECTLY_LED.value

    def is_open_ended_question(self):
        return self.question_type in [
            QuestionType.PRIMARY_OPEN_ENDED.value,
            QuestionType.PARAPHRASED_OPEN_ENDED.value,
        ]

    def is_primary_open_ended_question(self):
        return self.question_type == QuestionType.PRIMARY_OPEN_ENDED.value

    def is_paraphrased_open_ended_question(self):
        return self.question_type == QuestionType.PARAPHRASED_OPEN_ENDED.value

    def is_wrongly_led_question(self):
        return self.question_type == QuestionType.WRONGLY_LED.value


def load_questions_parquet(file_path: Path) -> list[ParquetFileRow]:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found")
    df = pd.read_parquet(file_path)
    return [ParquetFileRow(**row) for _, row in df.iterrows()]
