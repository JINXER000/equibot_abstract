"""Portable repository and external-dependency path resolution.

No absolute path is ever hardcoded in the codebase. The repository root is
located automatically (via the committed ``.repo_root`` marker, so the repo may
be renamed or vendored freely), and the locations of external checkouts -- which
live wherever the user cloned them -- are read from the environment through
:func:`env_path`.

Those external paths are populated automatically from a repository-local
``.env`` file (copied from ``.env.example``) at import time, so there is nothing
to ``export`` or ``source`` by hand. A real process-environment value always
wins over the ``.env`` file. An unset variable raises immediately at use time,
so a missing dependency fails loudly instead of resolving to a silently wrong
location.
"""

import os
from pathlib import Path

_ROOT_MARKER = ".repo_root"


def _find_repo_root():
    """Locate the repository root by walking up to the ``.repo_root`` marker.

    The common case (this module sits at ``<root>/equiv_primitive/policies/utils``) is
    three directories up; the upward walk is the fallback if the package is
    vendored at a different depth.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the {_ROOT_MARKER!r} repository marker above {here}."
    )


REPO_ROOT = _find_repo_root()


def _parse_value(raw):
    """Extract the value from the right-hand side of a ``.env`` assignment."""
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        quote = raw[0]
        end = raw.find(quote, 1)
        return raw[1:end] if end != -1 else raw[1:]
    # Unquoted: a whitespace-prefixed '#' begins an inline comment.
    comment = raw.find(" #")
    if comment != -1:
        raw = raw[:comment]
    return raw.strip()


def _load_dotenv():
    """Load ``REPO_ROOT/.env`` into the environment without overriding the shell."""
    dotenv = REPO_ROOT / ".env"
    if not dotenv.is_file():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key:
            os.environ.setdefault(key, _parse_value(value))


_load_dotenv()


def env_path(var):
    """Resolve an external-dependency path from the environment.

    Raises :class:`EnvironmentError` when ``var`` is unset, so misconfiguration
    is surfaced explicitly rather than defaulting to a wrong location.
    """
    value = os.environ.get(var)
    if not value:
        raise EnvironmentError(
            f"Required path variable {var!r} is not set. Copy .env.example to "
            f".env and fill in {var} (or export it in your shell)."
        )
    return os.path.expanduser(os.path.expandvars(value))
