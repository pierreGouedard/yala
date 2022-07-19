# Global import
from matplotlib import pyplot as plt


class PerfPlotter:

    def __init__(self, ax_x, ax_y, indices):
        self.x = ax_x
        self.y = ax_y
        self.indices = indices

    def __call__(self, ax_yhat):
        for i in range(ax_yhat.shape[1]):
            fig, (ax_got, ax_hat) = plt.subplots(1, 2)
            fig.suptitle(f'Viz GOT vs Preds #{i}')

            ax_got.scatter(self.x[self.y > 0, 0], self.x[self.y > 0, 1], c='r', marker='+')
            ax_got.scatter(self.x[self.y == 0, 0], self.x[self.y == 0, 1], c='b', marker='o')

            ax_hat.scatter(self.x[ax_yhat[:, i] > 0, 0], self.x[ax_yhat[:, i] > 0, 1], c='r', marker='+')
            ax_hat.scatter(self.x[ax_yhat[:, i] == 0, 0], self.x[ax_yhat[:, i] == 0, 1], c='b', marker='o')

            plt.show()
