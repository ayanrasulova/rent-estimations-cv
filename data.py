import pandas as pd
import numpy as np
import requests, os, re
from tqdm import tqdm

os.makedirs("images", exist_ok=True)

df = pd.read_csv("housing.csv")

# clean price (target)
df['price'] = df['price'].str.replace(r'[^\d]', '', regex=True).astype(float)
df = df[df['price'] > 0].copy()

# clean sqft 
df['sqft'] = df['sqft'].str.replace(r'[^\d]', '', regex=True)
df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')

#  clean scores (e.g. "81/100" → 81)
for col in ['walk_score', 'bike_score', 'transit_score']:
    df[col] = df[col].str.extract(r'(\d+)').astype(float)

# clean risk scores (e.g. "Minimal (1/10)" → 1) 
for col in ['flood_risk', 'fire_risk', 'wind_risk', 'air_risk', 'heat_risk']:
    df[col] = df[col].str.extract(r'\((\d+)/10\)').astype(float)

df = pd.get_dummies(df, columns=['region', 'property_type'], drop_first=True)

# drop rows
df = df.dropna(subset=['price', 'sqft', 'beds', 'baths'])

# download images
df['image_path'] = None
df['has_image'] = False

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
    save_path = f"images/{idx}.jpg"
    if os.path.exists(save_path):  # skip already downloaded
        df.at[idx, 'image_path'] = save_path
        df.at[idx, 'has_image'] = True
        continue
    try:
        r = requests.get(row['image_url'], timeout=10)
        if r.status_code == 200 and 'image' in r.headers.get('Content-Type', ''):
            with open(save_path, 'wb') as f:
                f.write(r.content)
            df.at[idx, 'image_path'] = save_path
            df.at[idx, 'has_image'] = True
    except Exception:
        pass

df = df[df['has_image'] == True].reset_index(drop=True)
df.to_csv("housing_clean.csv", index=False)
print(f"Saved {len(df)} rows with images to housing_clean.csv")