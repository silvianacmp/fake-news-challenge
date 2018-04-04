from abc import ABC, abstractmethod


class FeatureGenerator(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, articles, stances):
        pass
