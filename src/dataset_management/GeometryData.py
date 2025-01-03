from src.dataset_management.GeometryCalculator import GeometryCalculator


ENERGY_KEY = 'total_energy'

class GeometryData:

    def __init__(self, geometry, geometry_unit, elements, corresponding_data=None, units=None):
        self.geometry = geometry
        self.geometry_unit = geometry_unit
        self.elements = elements

        if corresponding_data is None or units is None:
            self.corresponding_data = {}
            self.corresponding_data_units = {}
        else:
            self.corresponding_data = corresponding_data
            self.corresponding_data_units = units

    def perform_calculations(self, geometry_calculator: GeometryCalculator):
        """
        Perform calculations on the geometry data.
        """
        geometry_calculator.calculate(self.geometry, self.elements)
        self.corresponding_data[ENERGY_KEY] = geometry_calculator.calculate(self.geometry, self.elements)
        self.corresponding_data_units[ENERGY_KEY] = geometry_calculator.get_unit()

    def get_geometries(self):
        return self.geometry

    def get_geometry_unit(self):
        return self.geometry_unit

    def get_elements(self):
        return self.elements

    def get_corresponding_data(self):
        return self.corresponding_data

    def get_corresponding_data_units(self):
        return self.corresponding_data_units

