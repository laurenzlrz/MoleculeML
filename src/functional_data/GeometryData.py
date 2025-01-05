from src.functional_data.GeometryCalculator import GeometryCalculator
from src.data_origins.AbstractMoleculeData import AbstractMoleculeData
from src.general.Property import Property
from src.data_origins.MD17MoleculeData import MD17Molecule

ENERGY_KEY = Property.TOTAL_ENERGY


class GeometryData:

    def __init__(self, molecule: AbstractMoleculeData, additional_attributes=None):
        self.geometry = molecule.getAttribute(Property.COORDINATES)
        self.geometry_unit = molecule.getUnit(Property.COORDINATES)
        self.elements = molecule.getAttribute(Property.ELEMENTS)
        self.element_unit = molecule.getAttribute(Property.ELEMENTS)

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
        attribute_arrays[Property.COORDINATES] = self.geometry
        attribute_arrays[Property.ELEMENTS] = self.elements
        attribute_units = self.additional_units.copy()
        attribute_units[Property.COORDINATES] = self.geometry_unit
        attribute_units[Property.ELEMENTS] = self.element_unit
        return MD17Molecule(str(self.elements), attribute_arrays, attribute_units)


