import faulthandler
import os

from tblite.ase import TBLite

from src.training.GeometrySchnetDB import GeometrySchnetDB
from src.functional_data.EnergyCalculator import EnergyCalculator
from src.functional_data.GeometryData import GeometryData
from src.data_origins.MD17DataLoader import MD17Dataloader
from src.functional_data.DifferenceProcessor import DifferenceProcessor
from src.functional_data.ChangeUnitProcessor import ChangeUnitProcessor

from src.general.Units import Units
from src.general.MolProperty import MolProperty

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEF_SELECTED_ATTRIBUTES = [MolProperty.TOTAL_ENERGY, MolProperty.OLD_ENERGIES, MolProperty.FORCES]
DEF_SELECTED_MOLECULES = ['aspirin']
DB_NAME = "Geometry5"


class AtomProcessPipeline:

    def __init__(self, db_name=DB_NAME):
        self.db_name = db_name
        self.molecule_data_objects = None
        self.molecule_geo_objects = None

    def process(self, selected_molecules=None, selected_attributes=None):
        if selected_attributes is None:
            selected_attributes = DEF_SELECTED_ATTRIBUTES
        if selected_molecules is None:
            selected_molecules = DEF_SELECTED_MOLECULES

        faulthandler.enable()
        MD17_loader = MD17Dataloader()
        MD17_loader.set_data_to_load()
        MD17_loader.set_random_split(99000, 10)

        MD17_loader.load_molecules()

        self.molecule_data_objects = MD17_loader.get_molecules(selected_molecules).values()

        self.molecule_geo_objects = [GeometryData(molecule, selected_attributes)
                                     for molecule in self.molecule_data_objects]

        self.calculate_on_geometries()

        # molecule_geometry_data.perform_calculations(geometry_calculator)
        geometrySchnetDBHandler = GeometrySchnetDB.create_new(DB_NAME, self.molecule_geo_objects,
                                                              overwrite_db=True)
        # geometrySchnetDBHandler.create_schnet_module()
        # print(geometrySchnetDBHandler)

    def calculate_on_geometries(self):
        geometry_calculator = EnergyCalculator(TBLite(method="GFN2-xTB"))
        [geometry_object.perform_calculations(geometry_calculator) for geometry_object in self.molecule_geo_objects]

        change_unit_processor = ChangeUnitProcessor(attribute=MolProperty.TOTAL_ENERGY_CALCULATED,
                                                    previous_unit=Units.EV,
                                                    result_unit=Units.KCALPERMOL,
                                                    factor=23.0621, atomwise=1)
        [change_unit_processor.calculate(geometry_object) for geometry_object in self.molecule_geo_objects]

        difference_processor = DifferenceProcessor(attribute1=MolProperty.TOTAL_ENERGY_TRUTH,
                                                   attribute2=MolProperty.TOTAL_ENERGY_CALCULATED,
                                                   result_attribute=MolProperty.TOTAL_ENERGY_DIFFERENCE)

        [difference_processor.calculate(geometry_object) for geometry_object in self.molecule_geo_objects]
