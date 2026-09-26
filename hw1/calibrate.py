import argparse
import csv
import json
import os

import numpy as np

from equations import flops, bytes_moved


def load_measurements(path):
    S, B, lat, mem, energy, is_val = [], [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row["S"])
            b = int(row["B"])
            is_oom = int(row["is_oom"])
            v = int(row["is_validation"])
            S.append(s)
            B.append(b)
            is_val.append(v)
            if is_oom:
                lat.append(np.nan)
                mem.append(np.nan)
                energy.append(np.nan)
            else:
                lat.append(float(row["latency_s"]))
                mem.append(float(row["memory_bytes"]))
                energy.append(float(row["energy_j"]) if row["energy_j"] else np.nan)
    return (np.array(S), np.array(B), np.array(lat), np.array(mem),
            np.array(energy), np.array(is_val, dtype=bool))


def fit_linear(y, X):
    theta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ theta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return theta, r2, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/measurements.csv")
    parser.add_argument("--output", default="results/theta.json")
    args = parser.parse_args()

    S, B, lat, mem, energy, is_val = load_measurements(args.input)

    train = ~np.isnan(lat) & ~is_val

    f = flops(S[train], B[train])
    b = bytes_moved(S[train], B[train])
    X = np.column_stack([np.ones_like(f), f, b])

    theta, r2_lat, lat_pred = fit_linear(lat[train], X)
    print("=== Latency model: t = θ₀ + θ₁·FLOPs + θ₂·bytes_moved ===")
    print(f"  θ₀ = {theta[0]:.6e}")
    print(f"  θ₁ = {theta[1]:.6e}  (s / FLOP)")
    print(f"  θ₂ = {theta[2]:.6e}  (s / byte)")
    print(f"  R² = {r2_lat:.6f}")

    val = ~np.isnan(lat) & is_val
    if val.any():
        f_val = flops(S[val], B[val])
        b_val = bytes_moved(S[val], B[val])
        X_val = np.column_stack([np.ones_like(f_val), f_val, b_val])
        lat_val_pred = X_val @ theta
        mape = np.mean(np.abs((lat[val] - lat_val_pred) / lat[val])) * 100
        print(f"  Validation MAPE = {mape:.2f}%")

    has_energy = ~np.isnan(energy)
    theta_e = None
    r2_energy = None
    train_e = has_energy & ~is_val
    if train_e.sum() >= 3:
        f_e = flops(S[train_e], B[train_e])
        b_e = bytes_moved(S[train_e], B[train_e])
        X_e = np.column_stack([np.ones_like(f_e), f_e, b_e])
        theta_e, r2_energy, _ = fit_linear(energy[train_e], X_e)
        print("\n=== Energy model: E = θₑ₀ + θₑ₁·FLOPs + θₑ₂·bytes_moved ===")
        print(f"  θₑ₀ = {theta_e[0]:.6e}")
        print(f"  θₑ₁ = {theta_e[1]:.6e}  (J / FLOP)")
        print(f"  θₑ₂ = {theta_e[2]:.6e}  (J / byte)")
        print(f"  R² = {r2_energy:.6f}")
    else:
        print("\n=== Energy model: not enough energy measurements ===")

    result = {
        "theta": theta.tolist(),
        "r2_latency": float(r2_lat),
    }
    if theta_e is not None:
        result["theta_energy"] = theta_e.tolist()
        result["r2_energy"] = float(r2_energy)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
