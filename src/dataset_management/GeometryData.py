from src.dataset_management.GeometryCalculator import GeometryCalculator


class GeometryData:

    def __init__(self, geometry, elements, corresponding_data=None):
        self.geometry = geometry
        self.elements = elements

        if corresponding_data is None:
            self.corresponding_data = []
        else:
            self.corresponding_data = corresponding_data

    def perform_calculations(self, geometry_calculator: GeometryCalculator):
        """
        Perform calculations on the geometry data.
        """
        geometry_calculator.calculate(self.geometry, self.elements)
        self.corresponding_data.append(geometry_calculator.calculate(self.geometry, self.elements))
