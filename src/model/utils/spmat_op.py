# Global import
from scipy.sparse import csc_matrix, lil_matrix
import numpy as np
from scipy.signal import convolve2d
from itertools import groupby
from operator import itemgetter


# TODO: last week: (Monday / Tuesday / Wednesday / Friday)
#   - Finalize this guy
#   - make sure the whole thing is working in casi-real life settings (test_hard)


def expand(sax_inputs, bitmap, n):
    ax_mask = (bitmap.b2f(sax_inputs.astype(int)).sum(axis=0).A[0, :] > 0)
    l_linds, l_cinds = [], []
    for i, sax_mask in enumerate(bitmap):
        if ax_mask[i]:
            ind_max, ind_min = sax_mask.indices.max(), sax_mask.indices.min()
            it = groupby(zip(*sax_inputs.multiply(sax_mask).nonzero()), itemgetter(1))
            for cind, l_sub_inds in it:
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))
                l_sub_linds = list(range(max(min(l_sub_linds) - n, ind_min), min(l_sub_linds))) + \
                    list(range(max(l_sub_linds) + 1, min(max(l_sub_linds) + n + 1, ind_max)))
                # Extend indices list
                l_linds.extend(l_sub_linds)
                l_cinds.extend([cind] * len(l_sub_linds))

    # Updat sp matrix
    sax_inputs = sax_inputs.tolil()
    sax_inputs[l_linds, l_cinds] = 1

    return sax_inputs.tocsc()


def shrink(sax_inputs, bitmap, n_shrink=0.4):

    ax_mask = (bitmap.b2f(sax_inputs.astype(int)).sum(axis=0).A[0, :] > 0)
    l_linds, l_cinds = [], []
    for i, sax_mask in enumerate(bitmap):
        if ax_mask[i]:
            it = groupby(zip(*sax_inputs.multiply(sax_mask).nonzero()), itemgetter(1))
            for cind, l_sub_inds in it:
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))
                l_sub_linds = list(range(min(l_sub_linds), min(l_sub_linds) + n_shrink)) + \
                    list(range(max(l_sub_linds) - n_shrink + 1, max(l_sub_linds) + 1))
                # Extend indices list
                l_linds.extend(l_sub_linds)
                l_cinds.extend([cind] * len(l_sub_linds))

    # Update sp matrix
    sax_inputs = sax_inputs.tolil()
    sax_inputs[l_linds, l_cinds] = 0

    return sax_inputs.tocsc()


def bounds(sax_inputs, bitmap):
    ax_mask = (bitmap.b2f(sax_inputs.astype(int)).sum(axis=0).A[0, :] > 0)
    l_linds, l_cinds = [], []
    for i, sax_mask in enumerate(bitmap):
        if ax_mask[i]:
            it = groupby(zip(*sax_inputs.multiply(sax_mask).nonzero()), itemgetter(1))
            for cind, l_sub_inds in it:
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))
                l_linds.extend([min(l_sub_linds), max(l_sub_linds)])
                l_cinds.extend([cind] * 2)

    # Update sp matrix
    sax_bound_inputs = lil_matrix(sax_inputs.shape)
    sax_bound_inputs[l_linds, l_cinds] = 1

    return sax_bound_inputs.tocsc()


def add_connex(sax_base, sax_inputs, bitmap):
    
    ax_mask = (bitmap.b2f(sax_inputs.astype(int)).sum(axis=0).A[0, :] > 0)
    l_linds, l_cinds = [], []
    for i, sax_mask in enumerate(bitmap):
        if ax_mask[i]:
            d_base_bounds = {
                c: (min(ls, key=itemgetter(1))[1], max(ls, key=itemgetter(1))[1])
                for c, ls in groupby(zip(*sax_base.multiply(sax_mask).nonzero()))
            }
            it = groupby(zip(*sax_inputs.multiply(sax_mask).nonzero()), itemgetter(1))
            for cind, l_sub_inds in it:
                # Get line indices
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))

                # Left bound
                cur_ind, k = d_base_bounds[cind][0], 1
                while k is not None:
                    if cur_ind - k in l_sub_inds:
                        l_linds.append(cur_ind - k)
                        k -= 1
                    else:
                        break

                # Right bound
                cur_ind, k = d_base_bounds[cind][1], 1
                while k is not None:
                    if cur_ind + k in l_sub_inds:
                        l_linds.append(cur_ind + k)
                        k += 1
                    else:
                        break

                l_cinds.extend([cind] * len(l_sub_linds))

    # Updat sp matrix
    sax_inputs = sax_inputs.tolil()
    sax_inputs[l_linds, l_cinds] = 1

    return sax_inputs.tocsc()