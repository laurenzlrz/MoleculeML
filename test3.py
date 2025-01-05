from src.functional_data.EnergyCalculator import EnergyCalculator
from src.functional_data.GeometryData import GeometryData
from src.data_origins.MD17DataLoader import MD17Dataloader

from tblite.ase import TBLite

MD17_loader = MD17Dataloader()
MD17_loader.set_data_to_load()
MD17_loader.set_random_split(99000, 10)
MD17_loader.load_molecules()

geometry_calculator = EnergyCalculator(TBLite(method="GFN2-xTB"))

molecule = MD17_loader.get_molecule('aspirin')
molecule_geometry_data = GeometryData(molecule)
molecule_geometry_data.perform_calculations(geometry_calculator)

# schnet_energy_geometry_data = EnergyGeometryData([molecule_geometry_data.get_geometries()])
# data_module = schnet_energy_geometry_data.get_schnet_module('first', pin_memory=False)
