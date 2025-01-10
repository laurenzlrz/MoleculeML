from src.functional_data.AbstractProcessor import AbstractProcessor
from src.general.MolProperty import MolProperty

UNIT_MISMATCH_MSG = "Unit of {attribute} must be {previous_unit}."
ELEMENTS_KEY = MolProperty.ELEMENTS

class ChangeUnitProcessor(AbstractProcessor):
    def __init__(self, attribute, previous_unit, result_unit, factor, atomwise=0):
        self.attribute = attribute
        self.previous_unit = previous_unit
        self.result_unit = result_unit
        self.factor = factor
        self.atomwise = atomwise

    def calculate(self, molecule_geo_object):
        array = molecule_geo_object.get_additional_attributes()[self.attribute]
        unit = molecule_geo_object.get_additional_units()[self.attribute]

        final_factor = self.factor
        if self.atomwise == 1:
            elements = len(molecule_geo_object.get_elements())
            final_factor *= elements

        if self.atomwise == -1:
            elements = len(molecule_geo_object.get_elements())
            final_factor *= (1/elements)

        if unit != self.previous_unit:
            raise ValueError(UNIT_MISMATCH_MSG.format(attribute=self.attribute, previous_unit=self.previous_unit))
        result_array = array * final_factor

        molecule_geo_object.replace_attribute(self.attribute, result_array, self.result_unit)
