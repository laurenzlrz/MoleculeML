from abc import ABC, abstractmethod

from src.general.props import NNDefaultValue
from src.training.GeometrySchnetDB import GeometrySchnetDB
from src.training.SchnetNN import AdditionSchnetNN, SchnetNN

DEF_CUTOFF = NNDefaultValue.DEF_CUTOFF
DEF_ATOM_BASIS_SIZE = NNDefaultValue.DEF_ATOM_BASIS_SIZE
DEF_NUM_OF_INTERACTIONS = NNDefaultValue.DEF_NUM_OF_INTERACTIONS
DEF_RBF_BASIS_SIZE = NNDefaultValue.DEF_RBF_BASIS_SIZE
DEF_LEARNING_RATE = NNDefaultValue.DEF_LEARNING_RATE


class ModelBuilder(ABC):

    @abstractmethod
    def load(self, db_loader: GeometrySchnetDB):
        pass

    @abstractmethod
    def build(self):
        pass


class SchnetNNBuilder(ModelBuilder):

    def __init__(self, additional_input_key_list, prediction_key_list,
                 atom_basis_size=DEF_ATOM_BASIS_SIZE, num_of_interactions=DEF_NUM_OF_INTERACTIONS,
                 rbf_basis_size=DEF_RBF_BASIS_SIZE, cut_off=DEF_CUTOFF, learning_rate=DEF_LEARNING_RATE):
        self.additional_input_key_list = additional_input_key_list
        self.prediction_key_list = prediction_key_list
        self.atom_basis_size = atom_basis_size
        self.num_of_interactions = num_of_interactions
        self.rbf_basis_size = rbf_basis_size
        self.cut_off = cut_off
        self.learning_rate = learning_rate

        self.additional_input_key_dict = None
        self.prediction_key_dict = None

    def load(self, db_loader: GeometrySchnetDB):
        """
        Models can automatically adapt to dataset shapes, therefore db_loader is needed to buildt
        """

        dim = db_loader.get_attribute_dimensions()
        self.additional_input_key_dict = {prop: dim[prop] for prop in self.additional_input_key_list}
        self.prediction_key_dict = {prop: dim[prop] for prop in self.prediction_key_list}

    def build(self):
        return SchnetNN(self.additional_input_key_dict, self.prediction_key_dict, self.atom_basis_size,
                        self.num_of_interactions, self.rbf_basis_size, self.cut_off, self.learning_rate)


class AdditionSchnetNNBuilder(SchnetNNBuilder):

    def __init__(self, additional_input_key_list, prediction_key_list, measure_keys, add1, add2, output,
                 atom_basis_size=DEF_ATOM_BASIS_SIZE, num_of_interactions=DEF_NUM_OF_INTERACTIONS,
                 rbf_basis_size=DEF_RBF_BASIS_SIZE, cut_off=DEF_CUTOFF, learning_rate=DEF_LEARNING_RATE):
        self.measure_keys = measure_keys
        self.add1 = add1
        self.add2 = add2
        self.output = output
        super().__init__(additional_input_key_list, prediction_key_list,
                         atom_basis_size, num_of_interactions, rbf_basis_size, cut_off, learning_rate)

    def build(self):
        return AdditionSchnetNN(self.additional_input_key_dict, self.prediction_key_dict, self.measure_keys,
                                self.add1, self.add2, self.output,
                                self.atom_basis_size, self.num_of_interactions, self.rbf_basis_size, self.cut_off,
                                self.learning_rate)
