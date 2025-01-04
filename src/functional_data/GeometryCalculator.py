from abc import ABC, abstractmethod


class GeometryCalculator(ABC):

    def __init__(self, unit, key):
        self.key = key
        self.unit = unit

    @abstractmethod
    def calculate(self, geometries, elements):
        pass

    def get_unit(self):
        return self.unit

    def get_key(self):
        return self.key
