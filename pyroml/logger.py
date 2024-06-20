from .utils import get_date


# TODO: incorporate Logger with logging package and/or rich(.progress?)
class Logger:
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def log(self, *args, force=False):
        if self.config.verbose or force:
            d = get_date()
            print(f"{d} | {self.name:<10} |", *args)
