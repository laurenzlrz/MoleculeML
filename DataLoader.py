import numpy as np
import pandas as pd

class Dataloader:

    def get_data(self):
        pass

ASPIRIN_PATH = "data/npz_data/rmd17_aspirin.npz"

class MD17Dataloader(Dataloader):

    def __init__(self):
        super().__init__()

    def get_data(self):
        data = np.load(ASPIRIN_PATH)
        print(data.files)
        print(data['coords'])


MD17_loader = MD17Dataloader()
frame = MD17_loader.get_data()
