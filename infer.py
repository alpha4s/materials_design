#!/usr/bin/env python3
import argparse
import joblib
import torch
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.neighbors import NearestNeighbors
from skopt import gp_minimize
from skopt.space import Real
from models import VAE, Predictor

# ───── Config ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIELDS     = [
    "band_gap","formation_energy_per_atom","energy_above_hull",
    "density","volume","nsites","energy_per_atom","total_magnetization"
]
LATENT_DIM = 20

# ───── Load Resources ─────────────────────────────────────────────────────────
scaler = joblib.load("scaler.pkl")
fea    = ElementProperty.from_preset("magpie")
df     = pd.read_csv("materials_dataset.csv").drop_duplicates("formula_pretty")

# Precompute feature & numeric matrices for NN lookups
X_feat = scaler.transform(
    np.stack([fea.featurize(Composition(f)) for f in df["formula_pretty"]]).astype(np.float32)
)
X_num  = df[FIELDS].values.astype(np.float32)

nn_feat = NearestNeighbors().fit(X_feat)
nn_num  = NearestNeighbors().fit(X_num)

# Load models
vae  = VAE(input_dim=X_feat.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
pred = Predictor(latent_dim=LATENT_DIM, output_dim=len(FIELDS)).to(DEVICE)
for m, path in [(vae, "vae_best.pth"), (pred, "predictor.pth")]:
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()


# ───── Helpers ────────────────────────────────────────────────────────────────
def find_neighbors(feats: np.ndarray, model: NearestNeighbors, k: int):
    dists, idxs = model.kneighbors(feats, n_neighbors=k)
    out = []
    for dr, ir in zip(dists, idxs):
        out.append([(df.iloc[i]["formula_pretty"], float(d))
                    for i, d in zip(ir, dr)])
    return out


# ───── Commands ──────────────────────────────────────────────────────────────

def cmd_predict(formula: str):
    raw = fea.featurize(Composition(formula))
    arr = scaler.transform([raw])
    x   = torch.from_numpy(arr).float().to(DEVICE)
    with torch.no_grad():
        _, mu, _ = vae(x)
        out = pred(mu).cpu().numpy().ravel()
    return dict(zip(FIELDS, out.tolist()))


def cmd_invert(args):
    targets = [getattr(args, f) for f in FIELDS]
    return find_neighbors(np.array([targets], dtype=np.float32), nn_num, args.k)[0]


def cmd_sample(args):
    z     = torch.randn(args.n, LATENT_DIM, device=DEVICE)
    feats = vae.decoder(z).cpu().detach().numpy()
    return find_neighbors(feats, nn_feat, args.k)


def cmd_design(args):
    tgt = torch.tensor([getattr(args, f) for f in FIELDS], device=DEVICE)
    z   = torch.randn(1, LATENT_DIM, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([z], lr=args.lr)
    for _ in range(args.steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(pred(z), tgt)
        loss.backward(); opt.step()
    feats = vae.decoder(z).cpu().detach().numpy()
    return find_neighbors(feats, nn_feat, args.k)[0]


def cmd_bo(args):
    # Bayesian-opt in latent space
    target = np.array([getattr(args, f) for f in FIELDS], dtype=np.float32)

    def objective(z):
        zt = torch.tensor(z, device=DEVICE).reshape(1, -1)
        with torch.no_grad():
            pred_out = pred(zt).cpu().numpy().ravel()
        return float(((pred_out - target) ** 2).mean())

    space = [Real(-3, 3) for _ in range(LATENT_DIM)]
    res   = gp_minimize(objective, space, n_calls=args.n_calls)

    z_opt = torch.tensor(res.x, device=DEVICE).reshape(1, -1)
    feats = vae.decoder(z_opt).cpu().detach().numpy()
    return find_neighbors(feats, nn_feat, args.k)[0]


# ───── CLI ───────────────────────────────────────────────────────────────────
def main():
    p  = argparse.ArgumentParser("Materials inference & design")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("predict").add_argument("--formula", required=True)

    inv = sp.add_parser("invert")
    [inv.add_argument(f"--{f}", type=float, required=True) for f in FIELDS]
    inv.add_argument("-k", type=int, default=3)

    samp = sp.add_parser("sample")
    samp.add_argument("-n", type=int, default=5)
    samp.add_argument("-k", type=int, default=3)

    des = sp.add_parser("design")
    [des.add_argument(f"--{f}", type=float, required=True) for f in FIELDS]
    des.add_argument("-k",    type=int,   default=3)
    des.add_argument("--lr",   type=float, default=0.1)
    des.add_argument("--steps",type=int,   default=500)

    bo = sp.add_parser("bo")
    [bo.add_argument(f"--{f}", type=float, required=True) for f in FIELDS]
    bo.add_argument("-k",      type=int,   default=3)
    bo.add_argument("--n_calls", type=int, default=50)

    args = p.parse_args()

    if args.cmd == "predict":
        print(cmd_predict(args.formula))

    elif args.cmd == "invert":
        for m, d in cmd_invert(args):
            print(f"{m} (dist {d:.4f})")

    elif args.cmd == "sample":
        for i, mats in enumerate(cmd_sample(args), 1):
            print(f"Sample {i}:")
            for m, d in mats:
                print(f"  {m} (dist {d:.4f})")

    elif args.cmd == "design":
        for m, d in cmd_design(args):
            print(f"{m} (dist {d:.4f})")

    else:  # bo
        for m, d in cmd_bo(args):
            print(f"{m} (dist {d:.4f})")


if __name__ == "__main__":
    main()
