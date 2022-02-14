# Global import
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy as copy

# Local import


class Tracker():
    tracking_infos = ['shape', 'area', 'n_no_gain']
    swap = {"compress": "expand", 'expand': 'compress'}

    def __init__(self, l_ids, tracker_params, n_features, min_area=1):
        # Completion criterion
        self.n_features = n_features
        self.min_area = min_area
        self.tracker_params = tracker_params

        # Attributes init
        self.indicator = {nid: [] for nid in l_ids}

    def update_tracker(self, nid, d_new):
        self.indicator[nid].append(d_new)
        return self

    def swap_criterion(self, area_change, shape_change):
        return area_change < self.tracker_params.min_area_gain and shape_change < self.tracker_params.min_shape_change

    def swap_components(self, components):
        for i, sub_comp in enumerate(components):
            d_new = sub_comp.partitions[0]
            d_prev = {} if not self.indicator[d_new['id']] else copy(self.indicator[d_new['id']][-1])

            if d_new['area'] < self.min_area:
                components.partitions[i]['stage'] = 'done'
                continue

            # Test whether the min precision and size gain is reached
            area_change = abs(d_new['area'] - d_prev.get('area', 0)) / d_prev.get('area', 1e-6)
            shape_change = abs(d_new['shape'] - d_prev.get('shape', np.zeros(self.n_features))).sum()

            if self.swap_criterion(area_change, shape_change):
                if d_prev['n_no_changes'] + 1 > self.tracker_params.max_no_changes:
                    components.partitions[i]['stage'] = 'done'

            self.update_tracker(d_new["id"], {k: d_new.get(k, d_prev.get(k, 0)) for k in self.tracking_infos})

        # Pop 'done' components
        i, stop = 0, False
        while not stop:
            comp = components[i]

            if comp.partitions[0]['stage'] == 'done':
                components.pop(i)
                stop = i >= len(components)
                continue

            i += 1
            stop = i >= len(components)

        return components

    def visualize_indicators(self):
        fig, (axe_prec, axe_size) = plt.subplots(1, 2)
        fig.suptitle(f'Metrics tracker viz')

        # Plot for each node
        for k, v in self.indicator.items():
            ax_x = np.arange(len(v))
            ax_precs, ax_sizes = np.array([d["shape"].sum() for d in v]), np.array([d["area"] for d in v])

            # Plot
            axe_prec.plot(ax_x, ax_precs, label=f"vertex {k}")
            axe_size.plot(ax_x, ax_sizes, label=f"vertex {k}")

            axe_prec.legend()
            axe_size.legend()

        plt.show()

    def get_complete_component(self):
        return [v[-1] for k, v in self.indicator.items() if v]
