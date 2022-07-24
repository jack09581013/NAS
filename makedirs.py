import os

os.makedirs('./images', exist_ok=True)

retrain_root = './models/retrain/'
os.makedirs(retrain_root + 'LEAStereo', exist_ok=True)
os.makedirs(retrain_root + 'Map2D', exist_ok=True)
os.makedirs(retrain_root + 'SimpleModel', exist_ok=True)
os.makedirs(retrain_root + 'EfficientNet_V2', exist_ok=True)