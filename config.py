"""Loader for config.yaml.

Settings live in config.yaml. This module reads it, expands ${VAR} and
${VAR:-default} from the environment, and re-exports every key as a module
attribute so call sites stay `import config as cfg; cfg.TOP_K`.
"""
import os
import re
from pathlib import Path

import yaml

_PATH = Path(__file__).with_name("config.yaml")
_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(text: str) -> str:
    return _ENV.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), text)


_settings = yaml.safe_load(_expand(_PATH.read_text()))
globals().update(_settings)
__all__ = list(_settings)
