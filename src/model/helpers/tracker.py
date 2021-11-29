# Global import
from matplotlib import pyplot as plt
import numpy as np

# Local import


class Tracker():
    tracking_infos = ['precision', 'n_firing', 'n_no_gain']

    def __init__(self, l_ids, min_firing, tracker_params):

        # Completion criterion
        self.min_firing = min_firing
        self.tracker_params = tracker_params

        # Attributes init
        self.indicator_tracker = {nid: [] for nid in l_ids}
        self.backup_components = {nid: None for nid in l_ids}

    def update_tracker(self, nid, d_new):
        self.indicator_tracker[nid].append(d_new)
        return self

    def pop_complete(self, components):

        l_complete_id, i, done = [], 0, False
        while not done:
            comp, prec_stop_criterion, size_stop_criterion = components[i], False, False
            d_new = comp.partitions[0]
            d_prev = {} if not self.indicator_tracker[d_new['id']] else self.indicator_tracker[d_new['id']][-1]
            # Test whether nb firing is high enough
            if d_new['n_firing'] < self.min_firing:
                self.update_tracker(d_new["id"], {k: d_new.get(k, d_prev.get(k, 0)) for k in self.tracking_infos})
                components.pop(i)
                done = i >= len(components)
                continue

            # Backup current component
            self.backup_components[d_new["id"]] = comp.copy()

            # Test whether the min precision and size gain is reached
            prec_gain = d_new['precision'] - d_prev.get('precision', 0)
            size_gain = (d_new['n_firing'] - d_prev.get('n_firing', 1)) / d_prev.get('n_firing', 1)
            prec_stop_criterion = prec_gain < self.tracker_params.min_prec_gain
            size_stop_criterion = size_gain < self.tracker_params.min_size_gain

            if size_stop_criterion and prec_stop_criterion:
                self.update_tracker(d_new["id"], {k: d_new.get(k, d_prev.get(k, 0) + 1) for k in self.tracking_infos})

                if d_prev['n_no_gain'] + 1 > self.tracker_params.max_no_gain:
                    components.pop(i)
                    done = i >= len(components)
                    continue
                # Update loop params
                i += 1
                done = i >= len(components)
                continue

            # Update tracker indicators and update loop params
            self.update_tracker(d_new["id"], {k: d_new.get(k, 0) for k in self.tracking_infos})
            i += 1
            done = i >= len(components)

        return components

    def visualize_indicators(self):
        fig, (axe_prec, axe_size) = plt.subplots(1, 2)
        fig.suptitle(f'Metrics tracker viz')

        # Plot for each node
        for k, v in self.indicator_tracker.items():
            ax_x = np.arange(len(v))
            ax_precs, ax_sizes = np.array([d["precision"] for d in v]), np.array([d["n_firing"] for d in v])

            # Plot
            axe_prec.plot(ax_x, ax_precs, label=f"vertex {k}")
            axe_size.plot(ax_x, ax_sizes, label=f"vertex {k}")

            axe_prec.legend()
            axe_size.legend()

        plt.show()

    def get_complete_component(self):
        return [x for x in self.backup_components.values() if x is not None]
