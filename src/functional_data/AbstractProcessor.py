from abc import ABC, abstractmethod

class AbstractProcessor(ABC):

    @abstractmethod
    def calculate(self, molecule_geo_object) -> None:
        pass


