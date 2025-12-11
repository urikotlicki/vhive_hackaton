from ABC import ABC, abstractmethod

class ModelRunner(ABC):
    @abstractmethod
    def pre_process(self, path: str):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def post_process(self):
        pass