# Global imports

# Local import
from src.model.helpers.amplifiers.base_amplifier import Amplifier


class RefineAmplifier(Amplifier):
    def __init__(
            self, server, fi_map, amplification_size, min_size=80, init_level=5, cover_thresh=0.5, size_thresh=0.5,
            max_precision=0.99, ci_value=0.8, gap_fill_length=2, debug=None
    ):
        super(RefineAmplifier, self).__init__(
            server, fi_map, amplification_size, min_size, init_level, cover_thresh, size_thresh, max_precision,
            ci_value, gap_fill_length, debug
        )

    def amplify(self, firing_graph, cmplt_comp=None):

        # Propagate signal
        amp_comp = self.compute_signals(firing_graph)

        # Select bits
        partial_comp, cmplt_comp = self.select_bits(amp_comp, cmplt_comp)

        # Build and return amplified firing_graph
        return partial_comp, cmplt_comp

    def select_bits(self, amp_comp, cmplt_comp=None):

        # Remove amplifier comp that are too small
        self.remove_unvalid_comp(amp_comp)

        if len(amp_comp) == 0:
            return None, cmplt_comp

        # Get feature map
        ax_feature_map = amp_comp.inputs.T.dot(self.fi_map).A.astype(int)

        # Select bit and get complete comp
        partial_comp, cmplt_comp = self.filter_complete_comp(super().select_bits(amp_comp), cmplt_comp, ax_feature_map)

        return partial_comp, cmplt_comp

    def remove_unvalid_comp(self, amp_comp):
        n_rm = 0
        for i in range(len(amp_comp)):
            if amp_comp.vertex_norm[i - n_rm] < self.min_size:
                amp_comp.pop(i - n_rm)
                n_rm += 1

    def filter_complete_comp(self, partial_comp, cmplt_comp, ax_feature_map):

        # Compute feature map after refinement
        ax_feature_map_new = partial_comp.inputs.astype(bool).T.dot(self.fi_map).A.astype(int)
        mask_complete = (ax_feature_map * ax_feature_map_new).sum(axis=1) == ax_feature_map.sum(axis=1)
        n_rm = 0

        # Filter out complete comp
        for i in range(len(partial_comp)):
            if mask_complete[i]:
                cmplt_comp += partial_comp.pop(i - n_rm)
                n_rm += 1

        return partial_comp if len(partial_comp) > 0 else None, cmplt_comp