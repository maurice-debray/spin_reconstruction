"""
Utilities for interacting with the local Git repository and reading TOML configuration files.

This module provides helpers to:
    - Retrieve the current working directory.
    - Check if the Git repository is clean (no uncommitted changes).
    - Retrieve the current commit hash (enforcing a clean state unless overridden).
    - Load configuration values from a TOML file, with or without a default fallback.
"""

import os
import tomllib
from typing import Any

from git import Repo


def get_cwd():
    """
    Get the current working directory.

    :returns: The absolute path of the current working directory.
    :rtype: str
    """
    return os.getcwd()


def check_dirty(path: str | None = None) -> bool:
    """
    Check whether the Git repository at the given path has uncommitted changes,
    including untracked files.

    :param str path: Optional. Path where to find the Git repo.
                     If None, uses the current working directory.
    :returns: True if the repository is dirty, False otherwise.
    :rtype: bool
    """
    if path is None:
        path = get_cwd()
    repo = Repo(path, search_parent_directories=True)
    return repo.is_dirty(untracked_files=True)


def get_commit_hash(path: str | None = None) -> str:
    """
    Retrieve the current commit hash of the Git repository at the given path.

    By default, the repository must be clean (no uncommitted or untracked changes).
    This check can be bypassed by setting the environment variable `UNSAFE_GIT_DIRTY=yes`.

    :param str path: Optional. Path to a directory inside the Git repository.
                     If None, uses the current working directory.
    :raises RuntimeError: If the repository is dirty and the environment variable
                          `UNSAFE_GIT_DIRTY` is not set to "yes".
    :returns: The SHA-1 hash of the current commit.
    :rtype: str
    """
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


def get_config(filename: str, config_path: list[str]) -> Any:
    """
    Load a nested configuration value from a TOML file.

    :param str filename: Path to the TOML config file without extension (``.conf.toml`` is appended).
    :param list[str] config_path: List of keys describing the nested path to the desired config value.
    :returns: The value found at the specified config path.
    :rtype: Any
    :raises KeyError: If a key in `config_path` is not present in the config file.
    """
    return get_config_default(filename, config_path, fail_missing=True)


def get_config_default(
    filename: str,
    config_path: list[str],
    default: Any = None,
    fail_missing: bool = False,
) -> Any:
    """
    Load a nested configuration value from a TOML file, returning a default if any key is missing.

    :param str filename: Path to the TOML config file without extension (``.conf.toml`` is appended).
    :param list[str] config_path: List of keys describing the nested path to the desired config value.
    :param Any default: Value to return if any key along the path is missing.
    :param bool fail_missing: Wether to raise a ``KeyError`` in case of missing key.
    :returns: The value found at the specified config path, or the default value.
    :rtype: Any
    :raises KeyError: If a key in `config_path` is not present in the config file and ``fail_missing`` is ``True``.
    """
    with open(f"{filename}.conf.toml", "rb") as f:
        data = tomllib.load(f)
    for c in config_path:
        if c not in data:
            if fail_missing:
                raise KeyError(
                    f"Key {c} not found in {filename}.conf.toml. Full path: {config_path}"
                )
            return default
        data = data[c]
    return data
