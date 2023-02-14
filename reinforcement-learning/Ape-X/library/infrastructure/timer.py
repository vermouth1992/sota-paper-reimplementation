import time


class StopWatch(object):
    def __init__(self, display='second'):
        super(StopWatch, self).__init__()
        self.start_time = None
        self.display = display
        assert display in ['second', 'minute', 'hour']
        self.logger = None

    def start(self):
        self.start_time = time.time()

    def lap(self):
        self.current_time = time.time()

    def log_tabular(self):
        self.lap()
        time_seconds = self.current_time - self.start_time
        if self.display == 'second':
            time_display = time_seconds
        elif self.display == 'minute':
            time_display = time_seconds / 60
        elif self.display == 'hour':
            time_display = time_seconds / 3600
        else:
            raise ValueError

        if self.logger is not None:
            self.logger.log_tabular(f'Time ({self.display})', time_display)
