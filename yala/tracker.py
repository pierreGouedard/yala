# Global import
from matplotlib import pyplot as plt
from numpy import arange, array, int32

# Local import
from yala.utils.data_models import FgComponents
from yala.firing_graph import YalaFiringGraph


class Tracker:

    def __init__(self, tracker_params, server):
        self.server = server
        self.tracker_params = tracker_params
        self.conv_comps = FgComponents.empty_comp()

        # Tracking
        self.tracking = {}

    def update_metrics(self, comps, update_tracking=False):
        # Get input
        sax_x = self.server.next_all_forward().sax_data_forward
        sax_y = self.server.next_all_backward().sax_data_backward

        # propagate through firing graph
        sax_fg = YalaFiringGraph.from_comp(comps).seq_propagate(sax_x)

        # Compute precision
        ax_areas = sax_fg.sum(axis=0).A[0]
        ax_precs = (sax_y.T.astype(int32).dot(sax_fg) / (sax_fg.sum(axis=0) + 1e-6)).A
        ax_precs, ax_labels = ax_precs.max(axis=0), ax_precs.argmax(axis=0)

        comps.partitions = [
            {**d, 'level': comps.levels[i], 'precision': ax_precs[i], 'label_id': ax_labels[i], 'area': ax_areas[i]}
            for i, d in enumerate(comps.partitions)
        ]

        if update_tracking:
            [self.update_tracking(part['id'], part) for part in comps.partitions]

        return self

    def update_tracking(self, cid, d_new_metrics):
        if not self.tracking.get(cid, {}):
            self.tracking[cid] = {'historic': []}

        self.tracking[cid]['area'] = d_new_metrics['area']
        self.tracking[cid]['precision'] = d_new_metrics['precision']
        self.tracking[cid]['level'] = d_new_metrics['level']
        self.tracking[cid]['historic'].append((d_new_metrics['area'], d_new_metrics['precision']))

        return self

    def refresh_tracking(self, comps):

        # Update metrics of comps
        self.update_metrics(comps)

        # Set stage of comps
        for i, comp in enumerate(comps):
            cid, d_new_metrics = comp.partitions[0]['id'], {**comp.partitions[0], 'level': comp.levels[0]}

            if d_new_metrics['area'] < self.tracker_params.min_area or comp.levels[0] == 1:
                comps.partitions[i]['stage'] = 'done'
                self.update_tracking(cid, d_new_metrics)
                continue

            # Test whether the area gain or level delta is not changing
            delta_area = abs(d_new_metrics['area'] - self.tracking[cid]['area']) / self.tracking[cid]['area']
            delta_level = abs(d_new_metrics['level'] - self.tracking[cid]['level'])

            if delta_area < self.tracker_params.min_diff_area and delta_level > 0:
                self.tracking[cid]['n_inactive'] = self.tracking[cid].get('n_inactive', 0) + 1

                if self.tracking[cid]['n_inactive'] > self.tracker_params.max_inactive:
                    comps.partitions[i]['stage'] = 'done'
            else:
                self.tracking[cid]['n_inactive'] = 0

            self.update_tracking(cid, d_new_metrics)

        # Pop 'done' comps
        return self.pop_conv_comp(comps)

    def pop_conv_comp(self, comps):
        i, stop = 0, False
        while not stop:
            if comps[i].partitions[0]['stage'] == 'done':
                self.conv_comps += comps.pop(i)
            else:
                i += 1

            stop = i >= len(comps)

        return comps

    def visualize_indicators(self):
        # Plot for each node
        for k, v in self.tracking.items():
            ax_x = arange(len(v['historic']))
            ax_areas = array([t[0] for t in v['historic']])
            plt.plot(ax_x, ax_areas, label=f"vertex {k}")

        plt.legend()
        plt.title(f'Metrics tracker viz')
        plt.show()
