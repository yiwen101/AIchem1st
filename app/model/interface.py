# define an interface, input video id and question, return the answer

from abc import ABC, abstractmethod
from app.model.structs import ParquetFileRow
class IVideoAgent(ABC):
    @abstractmethod
    def get_answer(self, row: ParquetFileRow) -> str:
        pass
    
    @abstractmethod
    def get_agent_name(self) -> str:
        pass

    def get_cleaned_answer(self, row: ParquetFileRow) -> str:
        answer = self.get_answer(row)
        # remove /n from answer
        answer = answer.replace("\n", "")
        '''
        # remove , from answer
        answer = answer.replace(",", "")
        '''
        return answer
