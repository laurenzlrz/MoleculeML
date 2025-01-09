from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

from src.general.Property import Property
from src.general.Units import Units


class AbstractMoleculeData(ABC):

    @abstractmethod
    def to2DStructuredNumpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def toDataFrame(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def toArrayDict(self) -> Dict[Property, np.ndarray]:
        pass

    @abstractmethod
    def addAttribute(self, array, attribute_key, attribute_unit):
        pass

    @abstractmethod
    def getAttribute(self, attribute: Property) -> np.array:
        pass

    @abstractmethod
    def getUnit(self, attribute: Property) -> Units:
        pass


