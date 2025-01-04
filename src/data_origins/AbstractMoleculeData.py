from abc import ABC, abstractmethod

from src.general.Property import Property


class AbstractMoleculeData(ABC):

    @abstractmethod
    def to2DStructuredNumpy(self):
        pass

    @abstractmethod
    def toDataFrame(self):
        pass

    @abstractmethod
    def getAttribute(self, attribute: Property):
        pass

    @abstractmethod
    def getUnit(self, attribute: Property):
        pass
