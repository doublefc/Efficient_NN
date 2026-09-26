# HW1 — Analytical Performance Model of a Small CNN

## GPU / Software

- **GPU:** Tesla T4, 16 GB, 70 W TDP
- **CUDA:** 12.2
- **PyTorch:** 2.2.0

## How to Reproduce

```bash
# 1. Install dependencies
pip install torch numpy matplotlib

# 2. Measure the network across the (S, B) grid
python measure.py                       # → results/measurements.csv

# 3. Calibrate theta from measurements
python calibrate.py                     # → results/theta.json

```

On Google Colab: open a GPU notebook, `!git clone <your-repo>`, then run the cells above.

## Model Architecture

| Layer | Config | Output size |
|-------|--------|-------------|
| Conv1 | 7×7, stride 2, 3→32 | B × 32 × S/2 × S/2 |
| MaxPool | 3×3, stride 2, pad 1 | B × 32 × S/4 × S/4 |
| Conv2 | 5×5, stride 1, 32→64 | B × 64 × S/4 × S/4 |
| Conv3 | 3×3, stride 2, 64→128 | B × 128 × S/8 × S/8 |
| Conv4 | 1×1, stride 1, 128→256 | B × 256 × S/8 × S/8 |
| Conv5 | 3×3, stride 2, 256→256 | B × 256 × S/16 × S/16 |
| Conv6 | 1×1, stride 1, 256→512 | B × 512 × S/16 × S/16 |
| Head | GAP → Linear 512→256 → ReLU → Linear 256→100 | B × 100 |

All convolutions use `padding = k // 2` and `bias = False`. After each conv: ReLU (inplace). Model runs in `eval()` mode, FP32.

## Analytical Equations

### FLOPs(S, B)

Convention: 1 multiply-accumulate = 2 FLOPs. MaxPool and ReLU contribute 0 FLOPs (comparisons only).

**FLOPs(S, B) = B × (17,716 × S² + 313,344)**

Breakdown per image (MACs):
| Layer | MACs |
|-------|------|
| Conv7×7 3→32 | 1,176 S² |
| Conv5×5 32→64 | 3,200 S² |
| Conv3×3 64→128 | 1,152 S² |
| Conv1×1 128→256 | 512 S² |
| Conv3×3 256→256 | 2,304 S² |
| Conv1×1 256→512 | 512 S² |
| GAP | 2 S² − 512 |
| Linear 512→256 | 131,072 |
| Linear 256→100 | 25,600 |
| **Total** | **8,858 S² + 156,672** |

### Memory(S, B)

Peak of `torch.cuda.max_memory_allocated()` during one forward pass in `inference_mode()`.

**Memory(S, B) = 4 × (1,040,324 + 11 × B × S²)**

- 1,040,324 = total model parameters (always resident)
- 11 × B × S² = peak activation elements (input 3BS² + first conv output 8BS² alive simultaneously)
- Factor 4 = bytes per FP32 element

### Bytes Moved (for latency & energy)

Total data transferred from DRAM (read input + read weights + write output per layer):

**BytesMoved(S, B) = 4 × (49 × B × S² + 1,636 × B + 1,040,324)**

### Latency(S, B, θ)

Linear model calibrated on GPU:

**Latency = θ₀ + θ₁ × FLOPs + θ₂ × BytesMoved**

- θ₀: constant overhead (kernel launches, synchronization)
- θ₁: time per FLOP (inverse of effective FLOP/s)
- θ₂: time per byte transferred (inverse of effective bandwidth)

### Energy(S, B, θₑ)

**Energy = θₑ₀ + θₑ₁ × FLOPs + θₑ₂ × BytesMoved**

## Results Summary

- **Latency R²** on training set: 0.985
- **Energy R²** on training set: 0.984
- **OOM configurations**: 0 out of 132 (T4 16 GB fits all tested configs)
- **Calibrated θ (latency):** θ₀ = −1.63e-02, θ₁ = −3.36e-11, θ₂ = 3.08e-09
- **Calibrated θₑ (energy):** θₑ₀ = −5.43e-01, θₑ₁ = −6.38e-10, θₑ₂ = 6.02e-08

## Discussion (1 page)

### Where the model works well

For moderate (S, B) where the arithmetic intensity is high enough that convolutions are compute-bound, the linear model captures latency well. Memory is predicted accurately because peak activations are dominated by the first layers' output tensors, which scale as B × S².

### Where the model breaks

1. **Launch-bound regime** (small S, B = 1): For tiny inputs, GPU kernels finish faster than the launch overhead. The linear model overestimates because θ₀ absorbs launch costs but the FLOPs/bytes terms don't vanish fast enough. Actual latency is dominated by kernel launch latency (~5–20 μs per kernel), not compute or memory.

2. **Memory-bound regime** (large B with moderate S): At large batch sizes, the convolutions become memory-bound (low arithmetic intensity). The bytes_moved term dominates, but the actual bandwidth depends on whether reads hit L2 cache. Our model assumes all reads come from DRAM at a fixed bandwidth, overestimating latency when caching is effective.

3. **OOM transitions**: Our memory equation predicts the OOM boundary as a hyperbola B × S² = const, but in practice cuDNN convolution algorithms use temporary workspace memory that varies with kernel choice. The allocator also has fragmentation overhead. So the true OOM boundary is slightly below our prediction.

4. **Non-linear batch scaling**: For very large B, latency doesn't scale perfectly linearly because GPU SM occupancy saturates — some kernels can't use more warps than the hardware provides. The linear model extrapolates poorly here.

5. **Energy measurement noise**: Energy estimated as power × latency is noisy because (a) nvidia-smi power readings are averaged over ~100 ms intervals, much longer than a single forward pass, and (b) GPU power has a large idle component that doesn't scale with workload.
