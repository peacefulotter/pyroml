import os


def set_bool_env(env: str, val: bool):
    os.environ[env] = "1" if val else "0"


def get_bool_env(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "True", "true", "t")
