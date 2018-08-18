import math
import time

class Statistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, loss=0.0, n_tokens=0):
        assert type(loss) is float or type(loss) is int
        assert type(n_tokens) is int
        self.loss = loss
        self.n_tokens = n_tokens
        self.start_time = time.time()

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        self.n_tokens += stat.n_tokens

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_tokens

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_tokens, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def clear(self):
        self.loss = 0.0
        self.n_tokens = 0
        self.start_time = time.time()
