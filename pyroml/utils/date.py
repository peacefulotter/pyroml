import time


def get_date() -> str:
    return time.strftime("%Y-%m-%d_%H:%M", time.gmtime(time.time()))
