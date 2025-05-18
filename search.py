# search.py
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from featurizers import featurize_composition

# load precomputed features & scaler
df     = pd.read_csv("materials_dataset.csv").drop_duplicates("formula_pretty")
scaler = joblib.load("scaler.pkl")

# build feature‐matrix (MagPie only)
X = scaler.transform(
    df["formula_pretty"].map(featurize_composition).tolist()
)
nn = NearestNeighbors().fit(X)

def find_neighbors(feats, k=3):
    """feats = list of feature‑vectors (unscaled)."""
    scaled = scaler.transform(feats)
    d,i = nn.kneighbors(scaled, n_neighbors=k)
    out = []
    for drow,irow in zip(d,i):
        out.append([(df.iloc[j]["formula_pretty"], float(dist))
                    for j,dist in zip(irow,drow)])
    return out
