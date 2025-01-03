from abc import ABC, abstractmethod

class MLData(ABC):

    def get_training_data(self):
        pass


class EnergyGeometryData(MLData):

    def __init__(self, geometry_objects):
        self.geometry_objects = geometry_objects
        self.geometry_data = []

    def get_training_data(self):
        pass