import sys
import shutil


def log_metric_averages(metrics, iterations):
    for metric_name, metric in metrics.items():
        average = metric.aggregate_value / iterations
        print(" {}: {:.4f}".format(metric_name, average), end="")


def log_metrics(metrics, inline=True):
    end = " " if inline else "\n"
    for metric in metrics:
        print("    {}: {:.4f}".format(metric, metrics[metric].current_value), end=end)
    print()


class ProgressBar:
    """Logs progress doing some task to the standard output.

    """
    def __init__(self, iterable=None):
        self.bar_width = shutil.get_terminal_size()[1]
        self.current = 0
        self.metrics = {}
        self.prev_block = 0

        if iterable is not None:
            self.iterator = iter(iterable)
            self.stop = iterable.stop

    def __call__(self, batch, num_batches):
        progress = batch / num_batches

        block = round(self.bar_width * progress)

        bar = "█" * block + " " * (self.bar_width - block)
        text = "\r   ▕{}▏ {:.0f}% ".format(bar, round(progress * 100, 0))

        sys.stdout.write(text)
        sys.stdout.flush()

        if self.prev_block < block:
            log_metric_averages(self.metrics, self.current - 1)
        self.prev_block = block

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        self.current += 1
        self.__call__(self.current, self.stop)
        return batch
