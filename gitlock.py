import os
import tomllib
from typing import Any

from git import Repo


def get_cwd():
    return os.getcwd()


def check_dirty(path=None):
    if path is None:
        path = get_cwd()
    repo = Repo(path, search_parent_directories=True)
    return repo.is_dirty(untracked_files=True)


def get_commit_hash(path=None):
    if path is None:
        path = get_cwd()
    repo = Repo(path, search_parent_directories=True)
    if os.environ.get("UNSAFE_GIT_DIRTY") != "yes" and repo.is_dirty(
        untracked_files=True
    ):
        raise RuntimeError(
            "The git repo is dirty. You must commit everything before running this software"
        )
    return repo.head.commit.hexsha


def get_config(filename, config_path) -> Any:
    with open(f"{filename}.conf.toml", "rb") as f:
        data = tomllib.load(f)
    for c in config_path:
        data = data[c]
    return data
