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


@dataclass
class YoutubeVideoInfo:
    title: str
    description: str
    transcript: str

@dataclass
class AttemptAnswerResponse:
    can_answer: bool
    answer: str
    reasoning: str

@dataclass
class ToolCall:
    tool_name: str
    parameters: dict

@dataclass
class QARecord:
    question: str
    answer: str
    reasoning: str
