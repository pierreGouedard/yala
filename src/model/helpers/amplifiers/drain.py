# Global imports
import numpy as np

# Local import
from src.model.patterns import YalaBasePatterns
from src.model.data_models import FgComponents
from src.model.helpers.amplifiers.base_amplifier import Amplifier


class DrainAmplifier(Amplifier):

    def __init__(
            self, server, fi_map, amplification_size, min_size=80, init_level=5, cover_thresh=0.5, size_thresh=0.5,
            max_precision=0.99, ci_value=0.8, gap_fill_length=0, level_margin=2, min_level_increment=True, debug=None
    ):
        self.level_margin = level_margin
        super(DrainAmplifier, self).__init__(
            server, fi_map, amplification_size, min_size, init_level, cover_thresh, size_thresh, max_precision,
            ci_value, gap_fill_length, min_level_increment, debug
        )

    def sample_and_amplify(self, n_sample=1):
        # Get randomly sampled input through servers and build partitions
        sax_inputs = self.server.get_random_samples(n_sample).T
        l_partitions = [{'label_id': 0, 'precision': 0} for _ in range(n_sample)]

        # Build sampled graph and amplify it
        fg = YalaBasePatterns.from_fg_comp(FgComponents(
            sax_inputs, l_partitions, np.array([self.init_level] * n_sample)
        ))

        partial_comp, cmplt_comp = self.amplify(fg, FgComponents.empty_comp())

        # Set init level
        partial_comp.levels = np.ones(len(partial_comp)) * self.init_level

        return partial_comp, cmplt_comp

    def amplify(self, firing_graph, cmplt_comp=None, l_pairs=None):

        # Propagate signal
        amp_comp = self.compute_signals(firing_graph)

        # Select bits
        partial_comp, cmplt_comp = self.select_bits(amp_comp, cmplt_comp, l_pairs)

        # Build and return amplified firing_graph
        return partial_comp, cmplt_comp

    def select_bits(self, amp_comp, cmplt_comp=None, l_pairs=None):

        if l_pairs is not None:
            cmplt_comp = self.filter_complete_comp(amp_comp, cmplt_comp, l_pairs)

        if self.debug is not None:
            if l_pairs is not None:
                self.debug = {'indices': l_pairs[self.debug['indices'][0]]}

        partial_comp = super().select_bits(amp_comp)

        if self.debug is not None:
            self.debug['indices'] = [self.debug['indices'][0]]

        return partial_comp, cmplt_comp

    def filter_complete_comp(self, amp_comp, cmplt_comp, l_pairs):
        n_rm = 0
        for i, l_pair in enumerate(l_pairs):
            norm_left, norm_right = amp_comp.vertex_norm[l_pair[0] - n_rm], amp_comp.vertex_norm[l_pair[1] - n_rm]
            p_left = amp_comp.partitions[l_pair[0] - n_rm]['precision']
            p_right = amp_comp.partitions[l_pair[1] - n_rm]['precision']

            if p_left > self.max_prec or p_right > self.max_prec:
                # Pop left and right part to build parent components
                left, n_rm = amp_comp.pop(l_pair[0] - n_rm), n_rm + 1
                right, n_rm = amp_comp.pop(l_pair[1] - n_rm), n_rm + 1

                cmplt_comp += FgComponents(
                    left.inputs + right.inputs, left.partitions,
                    (left.inputs + right.inputs).T.dot(self.fi_map).sum() - self.level_margin
                )

            elif norm_left < self.min_size and norm_right < self.min_size:
                # Pop left and right part to build parent components
                left, n_rm = amp_comp.pop(l_pair[0] - n_rm), n_rm + 1
                right, n_rm = amp_comp.pop(l_pair[1] - n_rm), n_rm + 1

                # Enrich completes components
                cmplt_comp += FgComponents(
                    left.inputs + right.inputs, left.partitions,
                    (left.inputs + right.inputs).T.dot(self.fi_map).sum() - self.level_margin
                )

            elif norm_left < self.min_size:
                # Pop left and get right child inputs
                left, n_rm = amp_comp.pop(l_pair[0] - n_rm), n_rm + 1

                # Enrich completes components
                cmplt_comp += FgComponents(
                    left.inputs, left.partitions, left.inputs.T.dot(self.fi_map).sum() - self.level_margin
                )

            elif norm_right < self.min_size:
                # Pop right and get left child inputs
                right, n_rm = amp_comp.pop(l_pair[1] - n_rm), n_rm + 1

                # Enrich completes components
                cmplt_comp += FgComponents(
                    right.inputs, right.partitions, right.inputs.T.dot(self.fi_map).sum() - self.level_margin
                )

        return cmplt_comp
