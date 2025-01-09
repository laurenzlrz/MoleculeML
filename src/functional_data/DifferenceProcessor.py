from src.functional_data.AbstractProcessor import AbstractProcessor
from src.functional_data.GeometryData import GeometryData

UNIT_MISMATCH_MSG = "Units of {unit1} and {unit2} must match."
DIMENSION_MISMATCH_MSG = "Dimensions of {dim1} and {dim2} must match."


class DifferenceProcessor(AbstractProcessor):
    def __init__(self, attribute1, attribute2, result_attribute):
        self.attribute1 = attribute1
        self.attribute2 = attribute2
        self.result_attribute = result_attribute

    def calculate(self, molecule_geo_object: GeometryData):
        unit1 = molecule_geo_object.get_additional_units()[self.attribute1]
        unit2 = molecule_geo_object.get_additional_units()[self.attribute2]
        if unit1 != unit2:
            raise ValueError(UNIT_MISMATCH_MSG.format(unit1=unit1, unit2=unit2))
        array1 = molecule_geo_object.get_additional_attributes()[self.attribute1]
        array2 = molecule_geo_object.get_additional_attributes()[self.attribute2]
        if array1.shape != array2.shape:
            raise ValueError(DIMENSION_MISMATCH_MSG.format(dim1=array1.shape, dim2=array2.shape))

        result_array = array1 - array2
        molecule_geo_object.add_attribute(self.result_attribute, result_array, unit1)
