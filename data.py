import requests
from PIL import Image
from io import BytesIO
import os
import pandas as pd

df = pd.read_csv("newyork_housing.csv")

os.makedirs("newyorkimages", exist_ok=True)
