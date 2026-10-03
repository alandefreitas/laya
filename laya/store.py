"""A directory of checkpoints served beside the built-in ones.

``laya-serve`` and ``laya-mcp-server`` register every checkpoint found in the store at startup, so a
fine-tune exported by the training recipe is served under its own name next to ``english``,
``multilingual`` and ``typed-decisions``: selected with ``model=<name>``, or without ``model=`` by
the question-id sets its manifest declares, and loaded, evicted and unloaded like the built-ins
(``LAYA_MAX_LOADED``, ``LAYA_IDLE_UNLOAD_SECONDS``).

Layout: one directory per checkpoint, named after it::

    ~/.local/share/laya/checkpoints/
        research/
            checkpoint.json          the manifest (required: it marks the checkpoint complete)
            model.safetensors        what the fine-tuning recipe exports
            rl_agent_config.json
            encoder/  tokenizer/

The store is ``$LAYA_CHECKPOINTS_DIR`` when set, else ``$XDG_DATA_HOME/laya/checkpoints``, else
``~/.local/share/laya/checkpoints``. A missing directory is an empty store.

The manifest, ``checkpoint.json``, is required, and it is the completion marker: write it last, once
the weights are in place. A directory holding weights but no manifest is not registered (it may be an
export still being written) and is reported as such. Its fields::

    {
      "name": "research",                 optional; defaults to the directory name
      "description": "Files claims from paper abstracts into a research brief",
      "source": "org/repo",               optional; a Hub repo id, or ["org/repo", "subfolder"],
                                          to load instead of this directory (a pushed checkpoint)
      "signatures": [["section"], {"section": "Which section of the brief does this claim belong to?",
                                   "subsection": "sha1:3f2a..."}],
                                          how a request selects this checkpoint when it names no
                                          model: a set of question ids (exact match), or a mapping
                                          from question id to the instructions text of that
                                          question (or its sha1: digest), which also requires the
                                          request to ask the same thing under that id; or a faceted
                                          object {"ids": [...], "instructions": {...},
                                          "options": {id: sorted choice labels or a score level count},
                                          "for": "what it is for"} where every facet present must
                                          match ("for" is documentation only; a mapping with an "ids"
                                          key that is not such an object is refused); several
                                          entries may name one checkpoint, one entry cannot name two
      "base": "english", "created": "2026-10-02", "training": {"rows": 22436, "epochs": 2}
                                          anything else is for the reader and is kept as is
    }

A directory without a manifest, and a directory whose manifest holds neither weights nor a ``source``,
is skipped; so is one whose manifest cannot be read or is invalid, whose name is a built-in's, an alias's
or ``auto``, that another directory already took, or whose signature another checkpoint already holds
(the directory sorted first keeps it). :func:`scan` returns the checkpoints that are fine together with
an ``errors`` map of the skipped ones (directory name -> why), which ``laya-serve`` logs and reports under
``registered_errors`` in ``/health``, so one bad checkpoint does not take the server down. A README or
other directory with no weights and no manifest is simply ignored. A store directory that exists but
cannot be listed raises, and so does a ``LAYA_DEFAULT_MODEL`` naming a skipped checkpoint, because both
are explicit configuration.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .router import DEFAULT_MODELS, _NAME_RE, canonical_name, parse_signature

MANIFEST = "checkpoint.json"
WEIGHTS = ("model.safetensors", "rl_agent_config.json")
ENV_DIR = "LAYA_CHECKPOINTS_DIR"


@dataclass
class Checkpoint:
    """One entry of the store: what to call it, where it is, and how it is selected."""

    name: str
    path: str
    source: Any
    signatures: List[Any] = field(default_factory=list)
    description: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)


def store_dir(environ: Optional[Dict[str, str]] = None) -> str:
    """The store directory: ``$LAYA_CHECKPOINTS_DIR``, else ``$XDG_DATA_HOME/laya/checkpoints``."""
    environ = os.environ if environ is None else environ
    raw = (environ.get("LAYA_CHECKPOINTS_DIR") or "").strip()
    if raw:
        return os.path.expanduser(raw)
    xdg = (environ.get("XDG_DATA_HOME") or "").strip()
    base = os.path.expanduser(xdg) if xdg else os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "laya", "checkpoints")


def discover(root: Optional[str] = None) -> Dict[str, Checkpoint]:
    """The checkpoints of the store that can be served, by name; see :func:`scan` for the ones that cannot."""
    return scan(root)[0]


def scan(root: Optional[str] = None,
         keep: Optional[Dict[str, Checkpoint]] = None) -> Tuple[Dict[str, Checkpoint], Dict[str, str]]:
    """``(checkpoints, errors)``: every servable checkpoint by name, and each skipped directory with why.

    A missing store is empty. A directory is skipped, not fatal, when it has no manifest but holds
    weights (the manifest is the completion marker), when its manifest is unreadable or invalid,
    when its name is taken (a built-in, an alias, ``auto``, or an earlier directory in sort order) or
    when one of its signatures is already another checkpoint's. A store directory that exists but cannot be
    listed raises ``ValueError``.

    ``keep`` maps names to the checkpoints already being served (a reload passes them): one that is
    unchanged on disk is admitted first, so a conflict with it refuses the new or edited manifest and
    never the working checkpoint, whatever the directory order. Without it, the directory sorted first wins.
    """
    root = store_dir() if root is None else os.path.expanduser(root)
    if not os.path.isdir(root):
        return {}, {}
    try:
        entries = sorted(os.listdir(root))
    except OSError as error:
        raise ValueError("cannot list the checkpoint store %s: %s" % (root, error)) from None
    found: Dict[str, Checkpoint] = {}
    errors: Dict[str, str] = {}
    signatures: Dict[str, List[Any]] = {}
    read: List[Tuple[bool, str, Checkpoint]] = []
    for entry in entries:
        path = os.path.join(root, entry)
        # `isdir` follows symlinks, so a checkpoint kept elsewhere can be linked into the store.
        if entry.startswith(".") or not os.path.isdir(path):
            continue
        try:
            checkpoint = _read(path, entry)
        except ValueError as error:
            errors[entry] = str(error)
            continue
        if checkpoint is not None:
            read.append((not (keep and keep.get(checkpoint.name) == checkpoint), entry, checkpoint))
    for _, entry, checkpoint in sorted(read, key=lambda item: item[0]):    # stable: unchanged first
        try:
            _admit(checkpoint, found, signatures)
        except ValueError as error:
            errors[entry] = str(error)
            continue
        found[checkpoint.name] = checkpoint
    return found, errors


def _admit(checkpoint: Checkpoint, found: Dict[str, Checkpoint], taken: Dict[str, List[Any]]) -> None:
    """Raise ValueError when `checkpoint` cannot be registered beside the ones already `found`."""
    name = checkpoint.name
    if name in DEFAULT_MODELS:
        raise ValueError("%s is named %r, which is a built-in checkpoint (or an alias of one); rename the "
                         "directory or set 'name' in its %s" % (checkpoint.path, name, MANIFEST))
    if name == "auto":
        raise ValueError("%s is named 'auto', which means automatic routing in every model= argument; "
                         "rename the directory or set 'name' in its %s" % (checkpoint.path, MANIFEST))
    if not _NAME_RE.match(name):
        raise ValueError("%s: invalid checkpoint name %r: use lowercase letters, digits, '.', '_' or '-', "
                         "starting with a letter or digit" % (checkpoint.path, name))
    if name in found:
        raise ValueError("the name %r is claimed by both %s and %s" % (name, found[name].path, checkpoint.path))
    parsed = [parse_signature(name, sig) for sig in checkpoint.signatures]
    for other, sigs in taken.items():
        for sig in parsed:
            if any(sig.conflicts(prior) for prior in sigs):
                raise ValueError("signature %s is already registered for %r or cannot be told apart from one "
                                 "of its signatures; a request must select one checkpoint only"
                                 % (sig.describe(), other))
    taken[name] = parsed


def _read(path: str, dirname: str) -> Optional[Checkpoint]:
    manifest_path = os.path.join(path, MANIFEST)
    has_weights = any(os.path.isfile(os.path.join(path, name)) for name in WEIGHTS)
    if not os.path.isfile(manifest_path):
        if has_weights:
            raise ValueError("no manifest: %s holds weights but no %s, so it is not registered (an export "
                             "still being written?); write the manifest last, once the files are complete"
                             % (path, MANIFEST))
        return None
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as error:
        raise ValueError("cannot read %s: %s" % (manifest_path, error)) from None
    if not isinstance(manifest, dict):
        raise ValueError("%s must hold a JSON object" % manifest_path)
    source = manifest.get("source")
    if source is None:
        if not has_weights:
            raise ValueError("%s: no weights beside the manifest and no 'source' to load from" % manifest_path)
        source = path
    elif isinstance(source, list):
        if len(source) != 2 or not all(isinstance(part, str) and part.strip() for part in source):
            raise ValueError("%s: 'source' must be a repo id or a [repo, subfolder] pair, got %r"
                             % (manifest_path, source))
        source = (source[0], source[1])
    elif not isinstance(source, str) or not source.strip():
        raise ValueError("%s: 'source' must be a repo id or a [repo, subfolder] pair, got %r"
                         % (manifest_path, source))
    name = manifest.get("name", dirname)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("%s: 'name' must be a non-empty string, got %r" % (manifest_path, name))
    signatures = manifest.get("signatures", [])
    if not isinstance(signatures, list) or not all(_valid_signature(sig) for sig in signatures):
        raise ValueError("%s: 'signatures' must be a list whose entries are non-empty lists of question ids or "
                         "non-empty objects mapping a question id to its instructions text, 'sha1:' digest or "
                         "null, or faceted {ids, instructions, options, for} objects, got %r" % (manifest_path, signatures))
    description = manifest.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("%s: 'description' must be a string, got %r" % (manifest_path, description))
    return Checkpoint(name=canonical_name(name), path=path, source=source,
                      signatures=[dict(sig) if isinstance(sig, dict) else list(sig) for sig in signatures],
                      description=description, manifest=manifest)


def _valid_signature(sig: Any) -> bool:
    if isinstance(sig, list):
        return bool(sig) and all(isinstance(q, str) and q.strip() for q in sig)
    if isinstance(sig, dict):
        if not (bool(sig) and all(isinstance(q, str) and q.strip() for q in sig)):
            return False
        try:
            parse_signature("", sig)     # the router's own reading: bad pins and digests are refused
        except (TypeError, ValueError):
            return False
        return True
    return False


def router_options(checkpoints: Dict[str, Checkpoint]) -> Dict[str, Any]:
    """``Router(...)`` keywords that register these checkpoints: ``models``, plus ``signatures`` when any declares one.

    Empty when the store is, so ``Router(**router_options(discover()))`` is the stock router then.
    """
    if not checkpoints:
        return {}
    options: Dict[str, Any] = {"models": {c.name: c.source for c in checkpoints.values()}}
    signatures = {c.name: c.signatures for c in checkpoints.values() if c.signatures}
    if signatures:
        options["signatures"] = signatures
    return options
