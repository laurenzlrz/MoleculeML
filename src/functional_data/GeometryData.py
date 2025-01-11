from src.functional_data.GeometryCalculator import GeometryCalculator
from src.data_origins.AbstractMoleculeData import AbstractMoleculeData
from src.general.props.MolProperty import MolProperty
from src.data_origins.MD17MoleculeData import MD17Molecule


ATTRIBUTE_ALREADY_EXISTING_MSG = "Attribute {attribute_key} already exists."
ATTRIBUTE_NOT_EXISTING_MSG = "Attribute {attribute_key} does not exist."

class GeometryData:

    def __init__(self, molecule: AbstractMoleculeData, additional_attributes=None):
        self.geometry = molecule.getAttribute(MolProperty.COORDINATES)
        self.geometry_unit = molecule.getUnit(MolProperty.COORDINATES)
        self.elements = molecule.getAttribute(MolProperty.ELEMENTS)
        self.element_unit = molecule.getAttribute(MolProperty.ELEMENTS)

        self.additional_attributes = {}
        self.additional_units = {}

        if additional_attributes is None:
            return

        for attribute in additional_attributes:
            self.additional_attributes[attribute] = molecule.getAttribute(attribute)
            self.additional_units[attribute] = molecule.getUnit(attribute)

    def perform_calculations(self, geometry_calculator: GeometryCalculator):
        """
        Perform calculations on the geometry data.
        """
        key = geometry_calculator.get_key()
        self.additional_attributes[key] = geometry_calculator.calculate(self.geometry, self.elements)
        self.additional_units[key] = geometry_calculator.get_unit()

    def add_attribute(self, attribute_key, attribute_array, attribute_unit):
        if attribute_key in self.additional_attributes.keys():
            raise KeyError(ATTRIBUTE_ALREADY_EXISTING_MSG.format(attribute_key=attribute_key))
        self.additional_attributes[attribute_key] = attribute_array
        self.additional_units[attribute_key] = attribute_unit

    def replace_attribute(self, attribute_key, attribute_array, attribute_unit=None):
        if attribute_key not in self.additional_attributes:
            raise KeyError(ATTRIBUTE_NOT_EXISTING_MSG.format(attribute_key=attribute_key))

        self.additional_attributes[attribute_key] = attribute_array
        if attribute_unit is not None:
            self.additional_units[attribute_key] = attribute_unit

    def get_geometries(self):
        return self.geometry

    def get_geometry_unit(self):
        return self.geometry_unit

    def get_elements(self):
        return self.elements

    def get_additional_attributes(self):
        return self.additional_attributes

    def get_additional_units(self):
        return self.additional_units

    def to_molecule(self):
        attribute_arrays = self.additional_attributes.copy()
        attribute_arrays[MolProperty.COORDINATES] = self.geometry
        attribute_arrays[MolProperty.ELEMENTS] = self.elements
        attribute_units = self.additional_units.copy()
        attribute_units[MolProperty.COORDINATES] = self.geometry_unit
        attribute_units[MolProperty.ELEMENTS] = self.element_unit
        return MD17Molecule(str(self.elements), attribute_arrays, attribute_units)
