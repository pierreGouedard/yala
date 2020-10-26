# Global import
import os
import pandas as pd
import pickle
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import diags, load_npz

# Local import
from settings import export_path


# Declare input and outputs
inputs = {
    'export_lazy': {'path': export_path, 'name': 'higgs/activation_lazy.npz'},
    'export_old': {'path': export_path, 'name': 'higgs/activation_old.npz'},
}
outputs = {
}

# Load activations
with open(os.path.join(inputs['export_old']['path'], inputs['export_old']['name']), 'rb') as handle:
    sax_activation_old = load_npz(handle)

with open(os.path.join(inputs['export_lazy']['path'], inputs['export_lazy']['name']), 'rb') as handle:
    sax_activation_lazy = load_npz(handle)

# Compute Tversky index as similarity measure
ax_sum = sax_activation_old.A.sum(axis=0, keepdims=True).repeat(sax_activation_old.shape[1], axis=0)
ax_sum += sax_activation_old.A.sum(axis=0, keepdims=True).transpose()
ax_inner = sax_activation_old.transpose().astype(int).dot(sax_activation_old).A

ax_sim_old = ax_inner / (ax_sum - ax_inner)
np.fill_diagonal(ax_sim_old, 0.)

print('Coverage old {}'.format((sax_activation_old.sum(axis=1).A > 0).sum()))
print('Average number very similar vertices old {}'.format((ax_sim_old > 0.9).sum(axis=0).mean()))

# Compute Tversky index as similarity measure
ax_sum = sax_activation_lazy.A.sum(axis=0, keepdims=True).repeat(sax_activation_lazy.shape[1], axis=0)
ax_sum += sax_activation_lazy.A.sum(axis=0, keepdims=True).transpose()
ax_inner = sax_activation_lazy.transpose().astype(int).dot(sax_activation_lazy).A

ax_sim_lazy = ax_inner / (ax_sum - ax_inner)
np.fill_diagonal(ax_sim_lazy, 0.)

print('Coverage lazy {}'.format((sax_activation_lazy.sum(axis=1).A > 0).sum()))
print('Average number very similar vertices lazy {}'.format((ax_sim_lazy > 0.9).sum(axis=0).mean()))

# Build 2D coordinates of vertices
ax_d = np.diag(1. / np.sqrt(ax_sim_old.sum(axis=1)))
ax_laplacian_old = ax_d.dot(ax_sim_old).dot(ax_d)
_, v = eigh(ax_laplacian_old)
ax_coord_old = v[:, -2:]

ax_d = np.diag(1. / np.sqrt(ax_sim_lazy.sum(axis=1)))
ax_laplacian_lazy = ax_d.dot(ax_sim_lazy).dot(ax_d)
w, v = eigh(ax_laplacian_lazy)
ax_coord_lazy = v[:, -2:]

plt.scatter(ax_coord_old[:, 0], ax_coord_old[:, 1], label="old")
plt.scatter(ax_coord_lazy[:, 0], ax_coord_lazy[:, 1], label="lazy")
plt.legend()
plt.show()
import IPython
IPython.embed()


