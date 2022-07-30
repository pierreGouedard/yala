# Global import
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy as copy

# Local import
from src.model.utils.data_models import FgComponents


class Tracker:
    tracking_infos = ['area', 'n_no_changes']
    swap = {"compress": "expand", 'expand': 'compress'}

    def __init__(self, l_ids, tracker_params, n_features, min_area=1):
        # Completion criterion
        self.n_features = n_features
        self.min_area = min_area
        self.tracker_params = tracker_params
        self.components = FgComponents.empty_comp()

        # Attributes init
        self.indicator = {nid: [] for nid in l_ids}

    def update_tracker(self, nid, d_new):
        self.indicator[nid].append(d_new)
        return self

    def swap_components(self, components):
        for i, sub_comp in enumerate(components):
            d_new = sub_comp.partitions[0]
            d_prev = {} if not self.indicator[d_new['id']] else copy(self.indicator[d_new['id']][-1])

            if d_new['area'] < self.min_area:
                components.partitions[i]['stage'] = 'done'
                continue

            # Test whether the min precision and size gain is reached
            delta_area = abs(d_new['area'] - d_prev.get('area', 0)) / d_prev.get('area', 1e-6)
            if delta_area < self.tracker_params.min_delta_area:
                d_new['n_no_changes'] = d_prev.get('n_no_changes', 0) + 1
                if d_prev.get('n_no_changes', 0) + 1 > self.tracker_params.max_no_changes:
                    components.partitions[i]['stage'] = 'done'
            else:
                d_new['n_no_changes'] = 0

            self.update_tracker(d_new["id"], {k: d_new.get(k, d_prev.get(k, 0)) for k in self.tracking_infos})

        # Pop 'done' components
        i, stop = 0, False
        while not stop:
            comp = components[i]

            if comp.partitions[0]['stage'] == 'done':
                components.pop(i)
                stop = i >= len(components)
                self.components += comp
                continue

            i += 1
            stop = i >= len(components)

        return components

    def visualize_indicators(self):
        # Plot for each node
        for k, v in self.indicator.items():
            ax_x = np.arange(len(v))
            ax_areas = np.array([d["area"] for d in v])
            plt.plot(ax_x, ax_areas, label=f"vertex {k}")

        plt.legend()
        plt.title(f'Metrics tracker viz')
        plt.show()
