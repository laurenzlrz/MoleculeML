from abc import ABC, abstractmethod


class AbstractDataLoader(ABC):

    @abstractmethod
    def load_molecules(self):
        pass

    @abstractmethod
    def get_molecules(self, molecule_names):
        pass
