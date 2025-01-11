import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.general.props.MolProperty import MolProperty
from src.training.GeometrySchnetDB import GeometrySchnetDB
from src.training.SchnetNN import SchnetNN

DB_NAME = "Geometry5"

DEF_PRED = [MolProperty.TOTAL_ENERGY]
DEF_SUPPLEMENTARY = [MolProperty.TOTAL_ENERGY_CALCULATED]


class TrainPipeline:

    def __init__(self, dbname=DB_NAME):
        self.dbname = dbname
        self.schnetDB = GeometrySchnetDB.load_existing(self.dbname)
        self.schnetNN = SchnetNN(self.schnetDB, DEF_SUPPLEMENTARY, DEF_PRED)

    def process(self):
        data_module = self.schnetDB.create_schnet_module(transforms=self.schnetNN.get_transforms())
        print(self.schnetDB)
        self.schnetNN.train(data_module)
        self.schnetNN.epoch_metrics_callback.save_to_csv()
        self.schnetNN.logger_saver_callback.save_to_csv()


