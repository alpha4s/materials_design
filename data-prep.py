from mp_api.client import MPRester
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm

API_KEY = ""  # Materials Project API key
CHUNK_SIZE = 1000
NUMERIC_FIELDS = [
    "band_gap", "formation_energy_per_atom", "energy_above_hull",
    "density", "volume", "nsites", "energy_per_atom", "total_magnetization"
]
ALL_FIELDS = ["formula_pretty"] + NUMERIC_FIELDS


def chunked(iterator, size):
    """Yield lists of up to `size` items from an iterator."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def fetch_and_featurize():
    featurizer = ElementProperty.from_preset("magpie")
    scaler = StandardScaler()

    # --- Pass 1: Fit scaler ---
    print("Fitting scaler")
    with MPRester(API_KEY) as m:
        docs = m.materials.summary.search(band_gap=(0, None), deprecated=False, fields=ALL_FIELDS)
        for batch in tqdm(chunked(docs, CHUNK_SIZE), desc="Pass 1 Fit"):
            feats = []
            for doc in batch:
                try:
                    feats.append(featurizer.featurize(Composition(doc.formula_pretty)))
                except:
                    continue
            arr = np.array(feats, dtype=np.float32)
            arr = arr[~np.isnan(arr).any(axis=1)]
            if arr.size:
                scaler.partial_fit(arr)

    joblib.dump(scaler, "scaler.pkl")
    print("scaler.pkl saved")

    # --- Pass 2: Generate datasets ---
    print("Creating datasets")
    dataset_csv = "materials_dataset.csv"
    if os.path.exists(dataset_csv):
        os.remove(dataset_csv)

    X_chunks, y_chunks = [], []
    with MPRester(API_KEY) as m:
        docs = m.materials.summary.search(band_gap=(0, None), deprecated=False, fields=ALL_FIELDS)
        for batch in tqdm(chunked(docs, CHUNK_SIZE), desc="Pass 2 Data"):
            comps, forms, labs = [], [], []
            for doc in batch:
                if not doc.formula_pretty:
                    continue
                try:
                    comps.append(Composition(doc.formula_pretty))
                    forms.append(doc.formula_pretty)
                    labs.append([getattr(doc, fld) for fld in NUMERIC_FIELDS])
                except:
                    continue
            if not comps:
                continue
            feats = np.array([featurizer.featurize(c) for c in comps], dtype=np.float32)
            mask = ~np.isnan(feats).any(axis=1)
            feats, labs, forms = feats[mask], np.array(labs, dtype=np.float32)[mask], np.array(forms)[mask]
            scaled = scaler.transform(feats)
            X_chunks.append(scaled)
            y_chunks.append(labs)

            df = pd.DataFrame(scaled, columns=featurizer.feature_labels())
            df.insert(0, "formula_pretty", forms)
            for i, fld in enumerate(NUMERIC_FIELDS, start=1):
                df.insert(i+1, fld, labs[:, i-1])
            df.to_csv(dataset_csv, mode="a", index=False, header=not os.path.exists(dataset_csv))

    X = np.vstack(X_chunks)
    y = np.vstack(y_chunks)
    np.savez("materials_data.npz", X=X, y=y)
    print(f"Saved {X.shape[0]} samples to materials_data.npz and {dataset_csv}")


if __name__ == "__main__":
    fetch_and_featurize()
