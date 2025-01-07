import os
from typing import List

import numpy as np
from ase import Atoms
from schnetpack.data import ASEAtomsData
from schnetpack.data import AtomsDataModule

from src.general.Property import Property
from src.functional_data.GeometryData import GeometryData
from src.general.Units import Units

DB_NAME_TEMPLATE = "schnet_db_{}.db"
DB_SPLIT_TEMPLATE = "schnet_split_{}.npz"
DB_PATH = "data/schnet_data/"

NO_GEO_OBJECTS_MSG = "No geometry objects to convert, therefore not possible to infer the units and properties."
UNITS_NOT_MATCHING = "Geometry units or property units of the different geometry objects do not match."
FILE_EXISTS_MSG = "Database {path} already exists, set overwrite_db to True to overwrite it."


class GeometrySchnetDB:

    @staticmethod
    def create_new(db_name, geometry_objects: List[GeometryData], overwrite_db=True):
        self = GeometrySchnetDB(db_name)
        if os.path.exists(self.path):
            if overwrite_db:
                os.remove(self.path)
            else:
                raise FileExistsError(FILE_EXISTS_MSG.format(path=self.path))

        self._create_not_existing_db(geometry_objects[0].get_geometry_unit(),
                                     geometry_objects[0].get_additional_units())
        self.add_data(geometry_objects)
        return self

    @staticmethod
    def load_existing(db_name):
        self = GeometrySchnetDB(db_name)
        self._load_existing_db()
        return self

    def __init__(self, db_name):
        self.db_name = db_name
        self.path = DB_PATH + DB_NAME_TEMPLATE.format(self.db_name)
        self.split_path = DB_PATH + DB_SPLIT_TEMPLATE.format(self.db_name)

        self.geometry_objects = []  # temporary
        self.geometry_unit = None
        self.prop_units = None

        self.schnet_db = None
        self.schnet_data_module = None

    def _load_existing_db(self):
        self.schnet_db = ASEAtomsData(self.path)
        self._load_units_from_db()

    def _create_not_existing_db(self, geo_unit, prop_units):
        """
        Geo_unit and prop_units are not loaded from class attributes, but are passed as arguments, because class
        enums are loaded from the schnet database, which is not yet created.

        :param geo_unit:
        :param prop_units:
        :return:
        """
        # Convert Enums into strings
        string_units = {key.value: value.value for key, value in prop_units.items()}
        schnet_db = ASEAtomsData.create(datapath=self.path, distance_unit=geo_unit.value,
                                        property_unit_dict=string_units)
        self.schnet_db = schnet_db
        self._load_units_from_db()

    def _load_units_from_db(self):
        """
        Loads the units from the schnet database into class units and transforms into the enum format.
        :return:
        """
        properties = [Property(prop) for prop in self.schnet_db.available_properties]
        self.prop_units = {prop: Units(self.schnet_db.units[prop.value]) for prop in properties}
        self.geometry_unit = Units(self.schnet_db.distance_unit)

    def _check_if_units_match(self, additional_geo_objects):
        if self.geometry_unit is None and self.prop_units is None:
            raise ValueError(NO_GEO_OBJECTS_MSG)

        for geometry_object in additional_geo_objects:
            assert_geo_unit = self.geometry_unit == geometry_object.get_geometry_unit()
            assert_prop_units = self.prop_units == geometry_object.get_additional_units()
            if not assert_geo_unit or not assert_prop_units:
                raise ValueError(UNITS_NOT_MATCHING)

    def add_data(self, geometry_objects: List[GeometryData]):
        self._check_if_units_match(geometry_objects)
        self.geometry_objects.extend(geometry_objects)
        atoms = []
        properties = []

        for geometry_object in geometry_objects:
            molecule_geometries = geometry_object.get_geometries()
            molecule_elements = geometry_object.get_elements()
            molecule_properties = geometry_object.get_additional_attributes()

            molecule_atom_list = [Atoms(numbers=molecule_elements, positions=geometry)
                                  for geometry in molecule_geometries]

            molecule_prop_list = [
                {prop.value: np.array([value[i]]) if not isinstance(value[i], np.ndarray) else value[i]
                 for prop, value in molecule_properties.items()}
                for i in range(len(molecule_geometries))]

            atoms.extend(molecule_atom_list)
            properties.extend(molecule_prop_list)

        # TODO remove
        print(atoms)
        print(properties)
        self.schnet_db.add_systems(property_list=properties, atoms_list=atoms)

    def get_schnet_db(self):
        return self.schnet_db

    def create_schnet_module(self, selected_properties=None, batch_size=2, num_train=2, num_val=2,
                             transforms=None, num_workers=1, pin_memory=True):
        if transforms is None:
            transforms = []

        # Only use selected properties and convert them from enums to strings
        if selected_properties is None:
            selected_unit_dict = {prop.value: prop_unit.value for prop, prop_unit in self.prop_units.items()}
        else:
            selected_unit_dict = {prop.value: self.prop_units[prop].value for prop in selected_properties}

        new_data_module = AtomsDataModule(self.path,
                                          distance_unit=self.geometry_unit.value,
                                          property_units=selected_unit_dict,
                                          load_properties=list(selected_unit_dict.keys()),
                                          batch_size=batch_size, num_train=num_train, num_val=num_val,
                                          transforms=transforms, num_workers=num_workers, pin_memory=pin_memory,
                                          split_file=self.split_path)
        new_data_module.prepare_data()
        new_data_module.setup()
        self.schnet_data_module = new_data_module
        return new_data_module

    def __str__(self):
        print_str = f"GeometrySchnetDB: {self.db_name} at {self.path}\n"
        print_str += f"Units in class: \n- Geo: {self.geometry_unit}\n- Props: {self.prop_units}\n"
        print_str += f"Lenght of geometry objects: {len(self.geometry_objects)}"

        total_length = 0
        geo_str = ""
        for i, geometry_object in enumerate(self.geometry_objects):
            geo_str += f"\nGeometry object {i} with elements: {geometry_object.get_elements()}\n"
            geo_str += f"Number of geometries: {len(geometry_object.get_geometries())}\n"
            total_length += len(geometry_object.get_geometries())

        print_str += geo_str
        print_str += f"Total number of geometries: {total_length}\n"

        if self.schnet_db is not None:
            print_str += "\nSchnetDB:\n"
            print_str += f"Number of entries: {len(self.schnet_db)}\n"
            print_str += f"Available properties: {self.schnet_db.available_properties}\n"
            print_str += f"Properties of molecule 0:\n"
            print_str += self.print_molecule(self.schnet_db[0])

        if self.schnet_data_module is not None:
            print_str += "\nSchnetDataModule:\n"
            print_str += f"Number of reference calculations: {len(self.schnet_data_module.dataset)}\n"
            print_str += f"Number of train data: {len(self.schnet_data_module.train_dataset)}\n"
            print_str += f"Number of validation data: {len(self.schnet_data_module.val_dataset)}\n"
            print_str += f"Number of test data: {len(self.schnet_data_module.test_dataset)}\n"
            print_str += f"Available properties:\n"
            for mol_property in self.schnet_data_module.dataset.available_properties:
                print_str += f"- {mol_property}\n"

            print_str += f"Properties of molecule 0 in dataset:\n"
            print_str += self.print_molecule(self.schnet_data_module.dataset[0])

            print_str += "\nValidation batch:\n"
            print_str += f"- Validation itself: {self.schnet_data_module.val_dataloader()}\n"
            print_str += f"- Number of batches: {len(self.schnet_data_module.val_dataloader())}\n"
            print_str += f"- Batches Properties\n"
            for batch in self.schnet_data_module.val_dataloader():
                print_str += f"- Batch: {batch.keys()}\n"
            print_str += f"Batches itself:\n"
            for batch in self.schnet_data_module.val_dataloader():
                print_str += f"- Batch: {batch}\n"
        return print_str

    @staticmethod
    def print_molecule(molecule):
        mol_str = ""
        for k, v in molecule.items():
            mol_str += f"- {k}: {v.shape}\n"
        return mol_str
