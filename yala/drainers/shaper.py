# Global import

# Local import
from yala.firing_graph import YalaFiringGraph
from yala.utils.data_models import FgComponents
from .drainer import YalaDrainer
from .utils import MaskManager
from yala.linalg.spmat_op import expand, shrink


class Shaper(YalaDrainer):
    """Shaper"""

    def __init__(self, server, drainer_params, plot_perf_enabled=False, visualizer=None):
        self.mask_manager = None
        self.plot_perf_enabled = plot_perf_enabled
        self.visualizer = visualizer

        # call parent constructor
        super().__init__(server, drainer_params)

    def shape(self, comps, n_expand=4, n_shrink=2):
        # Init
        self.setup_params(comps)
        self.mask_manager, conv_comps = MaskManager(comps, self.server.bitmap), FgComponents.empty_comp()

        # Core loop
        while len(comps) > 0:
            # Build inputs
            sax_curr_inputs = self.mask_manager.get_curr_bmask(comps.inputs)
            sax_base_inputs = shrink(sax_curr_inputs.copy(), self.server.bitmap, n_shrink)
            sax_other_inputs = self.mask_manager.get_oth_bmask(comps.inputs)
            sax_drain_inputs = expand(sax_base_inputs, self.server.bitmap, n_expand, keep_only_expanded=True)

            # Instantiate the firing graph
            self.firing_graph = YalaFiringGraph.from_inputs(
                sax_other_inputs + sax_drain_inputs, sax_drain_inputs, comps.levels,
                comps.partitions
            )

            # Drain
            comps = self.drain().select(sax_base_inputs, sax_other_inputs)

            # Check for convergence
            conv_comps_, comps = (
                self.mask_manager.update_counter(sax_curr_inputs, comps.inputs)
                .pop_no_changes(comps)
            )

            if not conv_comps_.empty:
                conv_comps += conv_comps_
                self.update_params(comps)

        # reset pattern backward of server
        self.server.pattern_backward = None
        self.server.sax_pattern_backward = None

        # Plot debug mode
        if self.plot_perf_enabled:
            self.visualizer.visualize_comp(conv_comps, self.server)

        return conv_comps
