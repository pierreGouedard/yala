# Global import
from scipy.sparse import lil_matrix
from itertools import groupby
from operator import itemgetter


def expand(sax_inputs, bitmap, n):
    ax_mask = (bitmap.b2f(sax_inputs.astype(int)).sum(axis=0).A[0, :] > 0)
    l_linds, l_cinds = [], []
    for i, sax_mask in enumerate(bitmap):
        if ax_mask[i]:
            ind_max, ind_min = sax_mask.indices.max(), sax_mask.indices.min()
            it = groupby(sorted(zip(*sax_inputs.multiply(sax_mask).nonzero()), key=itemgetter(1)), itemgetter(1))
            for cind, l_sub_inds in it:
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))
                l_sub_linds = list(range(max(min(l_sub_linds) - n, ind_min), min(l_sub_linds))) + \
                    list(range(max(l_sub_linds) + 1, min(max(l_sub_linds) + n + 1, ind_max + 1)))
                # Extend indices list
                l_linds.extend(l_sub_linds)
                l_cinds.extend([cind] * len(l_sub_linds))

    # Updat sp matrix
    sax_inputs = sax_inputs.tolil()
    sax_inputs[l_linds, l_cinds] = 1

    return sax_inputs.tocsc()


def shrink(sax_inputs, bitmap, n_shrink=2):
    ax_mask = (bitmap.b2f(sax_inputs.astype(int)).sum(axis=0).A[0, :] > 0)
    l_linds, l_cinds = [], []
    for i, sax_mask in enumerate(bitmap):
        if ax_mask[i]:
            it = groupby(sorted(zip(*sax_inputs.multiply(sax_mask).nonzero()), key=itemgetter(1)), itemgetter(1))
            for cind, l_sub_inds in it:
                # Get list of nonzero inds and set nb ind to shrink
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))
                n_sub_shrink = min(max(int((len(l_sub_linds) / 2) - 1), 0), n_shrink)

                # Get inds to shrink
                l_sub_linds = list(range(min(l_sub_linds), min(l_sub_linds) + n_sub_shrink)) + \
                    list(range(max(l_sub_linds) - n_sub_shrink + 1, max(l_sub_linds) + 1))
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
            it = groupby(sorted(zip(*sax_inputs.multiply(sax_mask).nonzero()), key=itemgetter(1)), itemgetter(1))
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

            # Init base bounds
            d_base_bounds = {}
            it = groupby(sorted(zip(*sax_base.multiply(sax_mask).nonzero()), key=itemgetter(1)), itemgetter(1))
            for c, it_sub_inds in it:
                l_sub_inds = [t[0] for t in it_sub_inds]
                d_base_bounds[c] = (min(l_sub_inds), max(l_sub_inds))

            it = groupby(sorted(zip(*sax_inputs.multiply(sax_mask).nonzero()), key=itemgetter(1)), itemgetter(1))
            for cind, l_sub_inds in it:
                # Get line indices
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))

                # Left bound
                cur_ind, k = d_base_bounds[cind][0], 1
                while k is not None:
                    if cur_ind - k in l_sub_linds:
                        l_linds.append(cur_ind - k)
                        l_cinds.append(cind)
                        k += 1
                    else:
                        break

                # Right bound
                cur_ind, k = d_base_bounds[cind][1], 1
                while k is not None:
                    if cur_ind + k in l_sub_linds:
                        l_linds.append(cur_ind + k)
                        l_cinds.append(cind)
                        k += 1
                    else:
                        break

    # Update sp matrix
    if l_linds:
        sax_base = sax_base.tolil()
        sax_base[l_linds, l_cinds] = 1

    return sax_base


def fill_gap(sax_inputs, bitmap):
    ax_mask = (bitmap.b2f(sax_inputs.astype(int)).sum(axis=0).A[0, :] > 0)
    l_linds, l_cinds = [], []
    for i, sax_mask in enumerate(bitmap):
        if ax_mask[i]:
            it = groupby(sorted(zip(*sax_inputs.multiply(sax_mask).nonzero()), key=itemgetter(1)), itemgetter(1))
            for cind, l_sub_inds in it:
                # Get all index between bounds
                l_sub_linds = list(map(itemgetter(0), l_sub_inds))
                l_sub_linds = list(range(min(l_sub_linds), max(l_sub_linds) + 1))

                # Extend list of indices to set to 1
                l_linds.extend(l_sub_linds)
                l_cinds.extend([cind] * len(l_sub_linds))

    # Update sp matrix
    sax_inputs = sax_inputs.tolil()
    sax_inputs[l_linds, l_cinds] = 1

    return sax_inputs.tocsc()
