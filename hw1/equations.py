import numpy as np


NUM_PARAMS = 1_040_324

A_MAC = 8858.0
B_MAC = 156672.0

A_FLOP = 2 * A_MAC
B_FLOP = 2 * B_MAC

PEAK_ACT_COEFF = 11.0

C_ELEM = 49.0
D_ELEM = 1636.0
E_ELEM = 1_040_324.0

BYTES_PER_ELEM = 4


def flops(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return B * (A_FLOP * S**2 + B_FLOP)


def memory(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    param_mem = NUM_PARAMS * BYTES_PER_ELEM
    act_mem = BYTES_PER_ELEM * PEAK_ACT_COEFF * B * S**2
    return param_mem + act_mem


def bytes_moved(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    elements = C_ELEM * B * S**2 + D_ELEM * B + E_ELEM
    return BYTES_PER_ELEM * elements


def latency(image_size, batch, theta):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    f = flops(S, B)
    b = bytes_moved(S, B)
    return theta[0] + theta[1] * f + theta[2] * b


def energy(image_size, batch, theta_energy):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    theta_energy = np.asarray(theta_energy, dtype=np.float64)
    f = flops(S, B)
    b = bytes_moved(S, B)
    return theta_energy[0] + theta_energy[1] * f + theta_energy[2] * b
