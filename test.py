import pandas as pd
import numpy as np
import requests, os, re
from tqdm import tqdm

df = pd.read_csv("listings_clean.csv")

print(df.head())