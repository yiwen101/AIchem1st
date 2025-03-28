from dataclasses import dataclass

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


