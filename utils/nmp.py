# Global import
import numpy as np
from scipy.sparse import save_npz, load_npz, vstack, hstack, coo_matrix

# Local import
from utils.driver import FileDriver


# Should inherit from FileDriver, that inherit from Driver
class NumpyDriver(FileDriver):

    def __init__(self,):
        self.d_stream = None
        FileDriver.__init__(self, 'numpy driver', 'Driver use to read / write any numpy arrays', streamable=True)

    def read_file(self, url, **kwargs):
        if kwargs.get('is_sparse', False):
            ax = load_npz(url).tocsc()
        else:
            ax = np.load(url)

        return ax

    def read_partitioned_file(self, url, is_sparse=False, l_files=None):

        if l_files is None:
            l_files = self.listdir(url)

        d_out = {}
        for filename in l_files:
            if is_sparse:
                d_out[filename.split('.')[0]] = self.read_file(self.join(url, filename), **{'is_sparse': True})
            else:
                d_out[filename.split('.')[0]] = self.read_file(self.join(url, filename))

        return d_out

    def write_file(self, ax, url, **kwargs):
        if kwargs.get('is_sparse', False):
            if not isinstance(ax, coo_matrix):
                ax = ax.tocoo()

            save_npz(url, ax)
        else:
            np.save(url, ax)

    def write_partioned_file(self, d_ax, url, is_sparse=False):

        for k, v in d_ax.items():
            if is_sparse:
                self.write_file(v, self.join(url, '{}.{}'.format(k, 'npz')), **{'is_sparse': True})
            else:
                self.write_file(v, self.join(url, '{}.{}'.format(k, 'npy')))

    def init_stream(self, url, is_sparse=False, is_cyclic=False, orient=None):
        # Initialize file and element cursor
        self.d_stream = {'orient': orient, 'is_sparse': is_sparse, 'url': url, 'is_cyclic': is_cyclic, 'step': 0,
                         'cache': self.read_file(url, is_sparse=is_sparse)}

        return self

    def init_stream_partition(self, url, key_partition=lambda x: x, n_cache=2, orient=None, is_sparse=False,
                              is_cyclic=False):
        """

        :param url:
        :param key_partition:
        :param n_cache:
        :param orient:
        :param is_sparse:
        :param is_cyclic:
        :return:
        """

        # Initialize file and element cursor
        d_stream = {'orient': orient, 'n_cache': n_cache, 'is_sparse': is_sparse, 'key': key_partition,
                    'url': url, 'is_cyclic': is_cyclic, 'step': 0, 'offset': 0, 'partitions': self.listdir(url)}

        # Sort list of partitions
        d_stream['partitions'] = sorted(d_stream['partitions'], key=key_partition)

        # Complete d_stream with cache
        self.d_stream = self.load_cache_stream(d_stream)

        # Support at most matrices stream
        assert self.d_stream['cache'].ndim < 3, 'streaming of numpy array only available for dimension les than 3'

        return self

    def stream_next(self):

        # Assert stream has been initiated
        assert self.d_stream is not None, 'Streaming has not been initiated'

        # Case no partition
        if 'partitions' not in self.d_stream:

            if self.d_stream['cache'].ndim == 1:
                if self.d_stream['step'] < self.d_stream['cache'].shape[0]:
                    next_ = self.d_stream['cache'][self.d_stream['step']]
                else:
                    return None

                self.d_stream['step'] += 1

                if self.d_stream['is_cyclic']:
                    self.d_stream['step'] = self.d_stream['step'] % self.d_stream['cache'].shape[0]

            elif self.d_stream['orient'] == 'columns':
                if self.d_stream['step'] < self.d_stream['cache'].shape[-1]:
                    next_ = self.d_stream['cache'][:, self.d_stream['step']]

                else:
                    return None

                self.d_stream['step'] += 1

                if self.d_stream['is_cyclic']:
                    self.d_stream['step'] = self.d_stream['step'] % self.d_stream['cache'].shape[-1]
            else:
                if self.d_stream['step'] < self.d_stream['cache'].shape[0]:
                    next_ = self.d_stream['cache'][self.d_stream['step'], :]
                else:
                    return None

                self.d_stream['step'] += 1

                if self.d_stream['is_cyclic']:
                    self.d_stream['step'] = self.d_stream['step'] % self.d_stream['cache'].shape[0]

            return next_

        # Case stream has ended
        if self.d_stream['cache'] is None:
            return None

        # Case 1d array
        elif self.d_stream['cache'].ndim == 1:
            next_ = self.d_stream['cache'][self.d_stream['step']]

            if self.d_stream['step'] + 1 >= len(self.d_stream['cache']):
                self.d_stream = self.load_cache_stream(self.d_stream)

            else:
                self.d_stream['step'] += 1

        # Case stream 2d array column by column
        elif self.d_stream['orient'] == 'columns':
            next_ = self.d_stream['cache'][:, self.d_stream['step']]

            if self.d_stream['step'] + 1 >= self.d_stream['cache'].shape[-1]:
                self.d_stream = self.load_cache_stream(self.d_stream)

            else:
                self.d_stream['step'] += 1

        # Case stream 2d array line by line
        else:
            next_ = self.d_stream['cache'][self.d_stream['step'], :]

            if self.d_stream['step'] + 1 >= self.d_stream['cache'].shape[0]:
                self.d_stream = self.load_cache_stream(self.d_stream)
            else:
                self.d_stream['step'] += 1

        return next_

    def load_cache_stream(self, d_stream):

        # Handle border stream
        if len(d_stream['partitions']) == d_stream['offset'] and not d_stream['is_cyclic']:
            d_stream['cache'] = None
            return d_stream

        elif len(d_stream['partitions']) == d_stream['offset'] and d_stream['is_cyclic']:
            d_stream['offset'] = 0

        # Cache files for stream
        d_stream['cache'] = self.read_partitioned_file(
            d_stream['url'],
            l_files=d_stream['partitions'][d_stream['offset']: d_stream['offset'] + d_stream['n_cache']],
            is_sparse=d_stream['is_sparse']
        )

        # Gather cache
        d_stream['cache'] = self.gather_cache(d_stream['cache'], d_stream['orient'], d_stream['is_sparse'],
                                              d_stream['key'])

        # Increment offset and reset step
        d_stream['offset'] += min(d_stream['n_cache'], len(d_stream['partitions']) - d_stream['offset'])
        d_stream['step'] = 0

        return d_stream

    @staticmethod
    def gather_cache(cache, orient, is_sparse, key):
        if is_sparse:
            if orient is None:
                cache = hstack([v for _, v in sorted(cache.items(),  key=lambda t: key(t[0]))])

            elif orient == 'columns':
                cache = hstack([v for _, v in sorted(cache.items(),  key=lambda t: key(t[0]))])

            else:
                cache = vstack([v for _, v in sorted(cache.items(), key = lambda t: key(t[0]))])

        else:
            if orient is None:
                cache = np.hstack((v for _, v in sorted(cache.items(), key=lambda t: key(t[0]))))

            elif orient == 'columns':
                cache = np.hstack((v for _, v in sorted(cache.items(),  key=lambda t: key(t[0]))))

            else:
                cache = np.vstack((v for _, v in sorted(cache.items(), key = lambda t: key(t[0]))))

        return cache



