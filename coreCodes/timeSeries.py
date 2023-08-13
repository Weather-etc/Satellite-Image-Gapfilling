import numpy as np
import georasters as gr
import glob

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

# load images from training set
path_list = glob.glob('../data/trainSet/*.tif')
imgs = map(gr.from_file, path_list)

for img in imgs:
    img.plot()
    exit()
