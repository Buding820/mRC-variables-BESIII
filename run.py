#!/usr/bin/env python3 

import os, sys 
src = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(src)

import numpy as np 
from read import hepmc 

if __name__ == "__main__":
    mtxt = hepmc()
    # mtxt.read_file_to_pkl("./muon.txt")
    # mtxt.read_pkl_events("./muon.pkl")
    mtxt.plot_csv("./mRC.csv")


