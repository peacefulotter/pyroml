import os
from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return f"pyro_{name}"


class PyroEnv(AutoName):
    VERBOSE = auto()


def set_bool_env(env: PyroEnv, val: bool):
    os.environ[env.value] = "1" if val else "0"


def get_bool_env(name: str | PyroEnv) -> bool:
    if isinstance(name, PyroEnv):
        name = name.value
    return os.getenv(name, "0").lower() in ("1", "True", "true", "t")
