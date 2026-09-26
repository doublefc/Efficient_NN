import argparse
import csv
import os
import subprocess
import time

import numpy as np
import torch

from models import SmallCNN

BASE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

WARMUP = 5
N_MEASURE = 20
NUM_VALID_S = 4
NUM_VALID_B = 3

rng = np.random.RandomState(42)


def generate_grid():
    all_multiples = [s for s in range(32, 513, 16) if s not in BASE_SIZES]
    valid_s = sorted(rng.choice(all_multiples, size=min(NUM_VALID_S, len(all_multiples)), replace=False).tolist())

    non_pow2 = [b for b in range(1, 257) if b not in BASE_BATCHES and (b & (b - 1)) != 0]
    valid_b = sorted(rng.choice(non_pow2, size=min(NUM_VALID_B, len(non_pow2)), replace=False).tolist())

    configs = []
    for s in BASE_SIZES:
        for b in BASE_BATCHES:
            configs.append((s, b, False))
    for s in valid_s:
        for b in BASE_BATCHES:
            configs.append((s, b, True))
    for s in BASE_SIZES:
        for b in valid_b:
            configs.append((s, b, True))
    for s in valid_s:
        for b in valid_b:
            configs.append((s, b, True))

    seen = set()
    unique = []
    for s, b, v in configs:
        key = (s, b)
        if key not in seen:
            seen.add(key)
            unique.append((s, b, v))
    return unique


def measure_gpu_power_w():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip()
        return float(out.split("\n")[0])
    except Exception:
        return None


def measure_one(model, S, B, device):
    try:
        x = torch.randn(B, 3, S, S, device=device, dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    try:
        for _ in range(WARMUP):
            with torch.inference_mode():
                _ = model(x)
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    latencies = []
    power_samples = []

    try:
        for _ in range(N_MEASURE):
            pw = measure_gpu_power_w()
            if pw is not None:
                power_samples.append(pw)

            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    peak_mem = torch.cuda.max_memory_allocated()
    median_lat = float(np.median(latencies))
    avg_power = float(np.mean(power_samples)) if power_samples else None
    energy_j = avg_power * median_lat if avg_power is not None else None

    return {
        "latency_s": median_lat,
        "memory_bytes": peak_mem,
        "energy_j": energy_j,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/measurements.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model = SmallCNN().to(device).eval()

    configs = generate_grid()
    print(f"Total configurations: {len(configs)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["S", "B", "latency_s", "memory_bytes", "energy_j", "is_oom", "is_validation"])

        for i, (S, B, is_val) in enumerate(configs):
            tag = "VAL" if is_val else "BASE"
            print(f"[{i+1}/{len(configs)}] S={S:4d} B={B:4d} ({tag}) ... ", end="", flush=True)

            result = measure_one(model, S, B, device)

            if result is None:
                print("OOM")
                writer.writerow([S, B, "", "", "", 1, int(is_val)])
            else:
                e_str = f"{result['energy_j']:.6f}" if result["energy_j"] is not None else ""
                print(f"lat={result['latency_s']*1e3:.3f}ms  mem={result['memory_bytes']/1e6:.2f}MB")
                writer.writerow([
                    S, B,
                    f"{result['latency_s']:.9f}",
                    f"{result['memory_bytes']}",
                    e_str,
                    0,
                    int(is_val),
                ])

            f.flush()

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
