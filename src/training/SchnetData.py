from abc import ABC, abstractmethod

from ase import Atoms

from schnetpack.data import ASEAtomsData
from schnetpack.data import AtomsDataModule

DB_NAME_TEMPLATE = "schnet_db_{}.db"
DB_PATH = "data/schnet_data/"
DISTANCE_UNIT = 'Ang'


class SchnetData(ABC):

    @abstractmethod
    def get_schnet_module(self, name, batch_size=32, num_train=1000, num_val=100,
                          transforms=None, num_workers=4, pin_memory=False):
        pass


class EnergyGeometryData(SchnetData):

    def __init__(self, geometry_objects):
        self.geometry_objects = geometry_objects
        self.establish_data()
        self.atoms = None
        self.properties = None
        self.geometry_unit = None
        self.prop_units = None
        self.schnet_data_set = None
        self.schnet_data_module = None

    def establish_data(self):
        self.atoms = []
        self.properties = []
        self.geometry_unit = None
        self.prop_units = None
        self.schnet_data_set = None

        for geometry_object in self.geometry_objects:

            if self.geometry_unit is None:
                self.geometry_unit = geometry_object.get_geometry_unit()
                self.prop_units = geometry_object.get_corresponding_data_units()
            else:
                assert_geo_unit = self.geometry_unit == geometry_object.get_geometry_unit()
                assert_prop_units = self.prop_units == geometry_object.get_corresponding_data_units()
                if not assert_geo_unit or not assert_prop_units:
                    raise ValueError("Geometry units or property units do not match.")

            molecule_geometries = geometry_object.get_geometry_data()
            molecule_elements = geometry_object.get_elements()
            properties = geometry_object.get_properties()

            molecule_atom_list = [Atoms(numbers=molecule_elements, positions=geometry)
                                  for geometry in molecule_geometries]

            molecule_prop_list = [{prop: value[i] for prop, value in properties.items()}
                                  for i in range(len(molecule_geometries))]

            self.atoms.extend(molecule_atom_list)
            self.properties.extend(molecule_prop_list)

    def get_schnet_dataset(self, name):
        path = DB_PATH + DB_NAME_TEMPLATE.format(name)
        new_dataset = ASEAtomsData(path, self.geometry_unit, self.prop_units)
        new_dataset.add_systems(self.properties, self.atoms)
        self.schnet_data_set = new_dataset
        return self.schnet_data_set

    def get_schnet_module(self, name, batch_size=32, num_train=1000, num_val=100,
                          transforms=None, num_workers=4, pin_memory=False):
        path = DB_PATH + DB_NAME_TEMPLATE.format(name)
        if transforms is None:
            transforms = []

        new_data_module = AtomsDataModule(path, distance_unit=self.geometry_unit, property_units=self.prop_units,
                                          batch_size=batch_size, num_train=num_train, num_val=num_val,
                                          transforms=transforms, num_workers=num_workers, pin_memory=pin_memory)
        new_data_module.prepare_data()
        new_data_module.setup()
        self.schnet_data_module = new_data_module
        return new_data_module

