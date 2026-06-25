import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS
import astropy.coordinates as coord
import astropy.units as u
from sklearn.tree import DecisionTreeClassifier
import training.py



class pedicting_stellar_class(object):
    
    def __init__(self, ra,dec,radius,telescope):
        self.ra = ra
        self.dec = dec
        self.radius = radius
        self.telescope = telescope
        
    # function to download the data which is the colors for now.   
    def data_call(self):
        # Example coordinates 10 05 54.6780857376 +31 22 27.476065848
        pos = coord.SkyCoord('10h05m54.678s', '+31d22m27.476s', frame='icrs')
        
        # Query photometric data (PhotoObj)
        xid = SDSS.query_region(pos, radius=1 * u.arcsec, photoobj_fields=['ra', 'dec', 'u', 'g', 'r', 'i', 'z'])
        
        return xid:
        
    def test_data(self):
        
        return:
    
    




# function to fit the machine learning code 

# function to fit the test data.