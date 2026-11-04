"""Sampled process RSS and descriptive latency statistics."""
import statistics
import threading
import time


class RSSSampler:
    def __init__(self, interval=0.01):
        import psutil
        self.process = psutil.Process()
        self.interval = interval
        self.peak = 0.0
        self.stop_event = threading.Event()

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss / 1024**2)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()


def summarize(values):
    values = sorted(values)
    if not values:
        return {}
    def percentile(q):
        index = (len(values)-1)*q
        lo = int(index)
        hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(index-lo)
    mean = statistics.mean(values)
    return dict(count=len(values), mean=mean, median=statistics.median(values),
                std=statistics.pstdev(values), p95=percentile(.95), p99=percentile(.99),
                fps=1000/mean if mean else None)
