"""Route a request to the Laya checkpoint best suited to it.

Three checkpoints, measured on a shared benchmark (17,416 questions, one T4, identical questions
per model -- see the repository's benchmark notebook):

  english          convaiinnovations/laya                421M  ModernBERT-large, 512 tokens
  multilingual     convaiinnovations/laya-multilingual   322M  mmBERT-base, 1024 tokens, 100+ langs
  typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
                                                                fine-tuned on the typed-decisions
                                                                workflows

Why routing is worth it -- accuracy by language family:

                      english   multilingual
  MASSIVE intent  en    0.783       0.657        <- English checkpoint wins
  MASSIVE intent  non-en 0.306      0.451
  XNLI            en    0.860       0.843
  XNLI            non-en 0.521      0.731        <- +21 points for multilingual
  English suites        0.684       0.619

The English checkpoint does not gently degrade off English, it collapses: on 20-option MASSIVE
intent it scores 0.100 on Hindi and 0.103 on Korean, against 0.050 for random guessing -- and it
reports high confidence while doing so (ECE 0.855 on Hindi). Script detection is therefore the
primary routing signal.

`typed-decisions` is never selected automatically unless you opt in with
`auto_task_detection=True` or pass `task="typed_decisions"`: it is fine-tuned on four specific
synthetic workflows and should not be a silent default.
"""
import gc
import hashlib
import inspect
import json
import os
import re
import threading
import time
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

from .confidence import apply_confidence_gate, check_min_confidence
from .hooks import (
    HookRegistry, PredictContext, aggregate_usage, compose_hooks, dispatch, normalise_hooks,
    validate_timeout,
)
from .hooks import _SKIP_DEFAULTS
from .lang import analyse

# The hub repo bundles all three checkpoints; only the requested subfolder is downloaded.
BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}

# The same checkpoints also live in their own repos, for anyone who prefers them.
STANDALONE_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}


def _repo_str(spec):
    """Human-readable id for a model spec: 'repo' or 'repo/subfolder'."""
    repo, sub = _split(spec)
    return "%s/%s" % (repo, sub) if sub else repo


def _split(spec):
    """Normalise a model spec to (repo_or_path, subfolder)."""
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None

# Aliases people are likely to type.
_ALIASES = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

# Question-id signatures of the four typed-decisions workflows, used only when
# auto_task_detection is enabled.
_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}

# What a caller may name a checkpoint of their own, once `canonical_name` has trimmed and
# lowercased it (so `Papers` is registered as `papers`): letters, digits, `.`, `_` and `-`, starting
# with a letter or digit. No `/`, because a Hub id or a path is a source, not a name.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def canonical_name(name: Any) -> str:
    """Trim, lowercase and resolve an alias, without asking whether the result names a checkpoint.

    The spelling half of :func:`normalise_name`. `Router.resolve` applies it and then checks the
    Router's own registry, which holds the built-in checkpoints plus any the caller registered;
    `normalise_name` applies it and checks the built-in table alone.
    """
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


class RouteDecision(dict):
    """The routing outcome: which model, why, and what was detected.

    Behaves as a dict so it serialises straight into an API response.
    """

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    """Canonical name of a built-in checkpoint, or ValueError.

    This is the registry of the checkpoints the package ships. A Router may know more -- the
    checkpoints registered on it with `Router(models=...)` or `Router.register` -- and resolves
    names through `Router.resolve`, which accepts those too. Code with no Router at hand (the CLI's
    `--model`, `laya.load`) keeps using this one.
    """
    key = canonical_name(name)
    if key not in DEFAULT_MODELS:
        raise ValueError("unknown model %r; choose one of %s (or an alias: %s)"
                         % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES)))
    return key


def resolve_model_spec(name: str) -> Optional[Tuple[str, Optional[str]]]:
    """Registry spec for a checkpoint name or alias, or None when it is not one.

    The non-raising sibling of :func:`normalise_name`, for callers that also accept
    things the registry knows nothing about -- a Hub repo id, a local directory, an
    ONNX export. Those pass through untouched; a name or alias the registry does know
    resolves to its ``(repo, subfolder)`` pair, so ``load("typed-decisions")`` and
    ``Router(model="typed-decisions")`` name the same checkpoint from one table.
    """
    try:
        key = normalise_name(name)
    except ValueError:
        return None
    return tuple(_split(DEFAULT_MODELS[key]))


_MISSING: Any = object()          # a name absent from `models`, as opposed to registered with source None
_NOT_COMPUTED: Any = object()    # `match_signature(workflow=...)`: None already means "computed, no match"

_DIGEST_PREFIX = "sha1:"
_DIGEST_RE = re.compile(r"^sha1:[0-9a-f]{40}$", re.IGNORECASE)


def instructions_digest(instructions: Any) -> Optional[str]:
    """`sha1:<hex>` of a question's instructions, the form a signature pins them in; None for no instructions.

    Text is compared with its whitespace collapsed, so a reflowed prompt is the same question.
    Structured instructions (a dict or list) are compared as canonical JSON. A value that already
    carries the `sha1:` prefix is returned as it is (lowercased), so a manifest may hold either the
    text or the digest the decider logged; a `sha1:` value that is not 40 hex digits raises ValueError. The model never sees a question's id, only its instructions and
    options, which is why the instructions are what ties a request to the checkpoint trained on it.
    """
    if instructions is None:
        return None
    if isinstance(instructions, str):
        text = instructions.strip()
        if text[:len(_DIGEST_PREFIX)].lower() == _DIGEST_PREFIX:
            if not _DIGEST_RE.match(text):
                raise ValueError("malformed digest %r: expected 'sha1:' and 40 hex digits" % (text,))
            return text.lower()
        canonical = " ".join(text.split())
    else:
        canonical = json.dumps(instructions, sort_keys=True, ensure_ascii=False, default=str)
    return _DIGEST_PREFIX + hashlib.sha1(canonical.encode("utf-8")).hexdigest()


_FACETS = ("ids", "instructions", "options", "for")


def _is_described(raw: Any) -> bool:
    """True for the faceted shape `Signature.describe` reports: `ids` plus any of `instructions`, `options`, `for`."""
    return (isinstance(raw, MappingABC) and "ids" in raw and set(raw) <= set(_FACETS)
            and isinstance(raw["ids"], list)
            and isinstance(raw.get("instructions", {}), MappingABC)
            and isinstance(raw.get("options", {}), MappingABC))


def options_key(question: Any) -> Any:
    """What an `options` pin compares: the sorted choice labels, or the number of score levels.

    A `choice` question's `criteria` is a mapping label -> description; only the labels count, so
    rewording a description does not invalidate a checkpoint while adding or removing a label does.
    A `choice` question may also send its labels as a plain list; they count the same way. A `score`
    question's `criteria` is a list of level descriptions, and the count is what the head was trained
    on. A `noul` question has no options to pin; it answers None, which never matches. A question
    without a `type` is read by the shape of its criteria.
    """
    if not isinstance(question, MappingABC):
        return None
    criteria, kind = question.get("criteria"), question.get("type")
    if kind == "noul":
        return None
    if kind == "choice" and isinstance(criteria, (list, tuple)):
        return sorted(str(label) for label in criteria)
    if isinstance(criteria, MappingABC):
        return sorted(str(label) for label in criteria)
    if isinstance(criteria, (list, tuple)):
        return len(criteria)
    return None


class Signature:
    """One way a request selects a registered checkpoint without naming it.

    A signature is a set of facets, and every facet present must match. `ids` is always there: the
    exact set of question ids the request must carry. `pins` maps some of those ids to the `sha1:`
    digest of the instructions the question must carry under that id. `options` maps some of them
    to the answer set the question must offer, as `options_key` reads it: the sorted choice labels,
    or the number of score levels. Which facets a checkpoint pins says what it can survive: a kind
    whose answers never change pins its options too, and a changed label set then falls back to the
    base model instead of being answered by a checkpoint that never saw the new labels; a kind whose
    answers change every batch (the sections of a growing tree) pins instructions only, because the
    checkpoint learned to read option descriptions. `purpose` is free text for the reader and takes
    no part in matching.

    The signature with the most pinned facets is tried first, so two checkpoints may share an id set
    and be told apart by what they ask; two signatures on one id set whose facets neither contain
    nor contradict each other cannot belong to two checkpoints.
    """

    __slots__ = ("ids", "pins", "options", "purpose")

    def __init__(self, ids: Iterable[str], pins: Optional[Dict[str, str]] = None,
                 options: Optional[Dict[str, Any]] = None, purpose: Optional[str] = None):
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Iterable):
            raise TypeError("Signature ids must be an iterable of question ids, not %r" % (ids,))
        self.ids: FrozenSet[str] = frozenset(ids)
        self.pins: Dict[str, str] = dict(pins or {})
        self.options: Dict[str, Any] = {str(q): _canonical_options(v) for q, v in (options or {}).items()}
        self.purpose: Optional[str] = purpose

    def matches(self, questions: Dict[str, Any]) -> bool:
        if frozenset(questions) != self.ids:
            return False
        for qid, digest in self.pins.items():
            question = questions.get(qid)
            asked = question.get("instructions") if isinstance(question, MappingABC) else None
            try:
                if instructions_digest(asked) != digest:
                    return False
            except ValueError:      # the request's own text looks like a malformed digest: no match
                return False
        for qid, want in self.options.items():
            if options_key(questions.get(qid)) != want:
                return False
        return True

    def _facets(self) -> Dict[Tuple[str, str], Any]:
        facets: Dict[Tuple[str, str], Any] = {("instructions", q): d for q, d in self.pins.items()}
        facets.update({("options", q): _hashable(v) for q, v in self.options.items()})
        return facets

    def conflicts(self, other: "Signature") -> bool:
        """True when some request matches both and neither is the more specific claim.

        The ids must be equal and no shared facet may contradict. If one signature's facets strictly
        contain the other's, that one is more specific and is tried first, so there is no conflict.
        """
        if self.ids != other.ids:
            return False
        mine, theirs = self._facets(), other._facets()
        if any(theirs[k] != v for k, v in mine.items() if k in theirs):
            return False
        a, b = mine.items(), theirs.items()
        return not (a < b or b < a)

    def specificity(self) -> int:
        """How many facets beyond the ids are pinned; the most specific signature is tried first."""
        return len(self.pins) + len(self.options)

    def describe(self) -> Any:
        """The serialisable form `registered` reports: a sorted id list, or `ids` plus the facets present."""
        if not self.pins and not self.options and self.purpose is None:
            return sorted(self.ids)
        out: Dict[str, Any] = {"ids": sorted(self.ids)}
        if self.pins:
            out["instructions"] = dict(sorted(self.pins.items()))
        if self.options:
            out["options"] = dict(sorted(self.options.items()))
        if self.purpose is not None:
            out["for"] = self.purpose
        return out

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Signature) and self.ids == other.ids and self.pins == other.pins
                and self.options == other.options)

    def __hash__(self) -> int:
        return hash((self.ids, tuple(sorted(self._facets().items()))))

    def __repr__(self) -> str:
        return "Signature(ids=%r, pins=%r, options=%r)" % (sorted(self.ids), self.pins, self.options)


def _canonical_options(value: Any) -> Any:
    """An `options` pin as `options_key` would produce it: a sorted list of labels, or a level count."""
    if isinstance(value, bool) or not isinstance(value, (int, list, tuple)):
        raise TypeError("an options pin must be a list of choice labels or a score level count, got %r" % (value,))
    if isinstance(value, int):
        if value < 1:
            raise ValueError("a score level count must be positive, got %r" % (value,))
        return value
    if not value or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError("an options pin must list non-empty labels, got %r" % (value,))
    if len(set(value)) != len(value):
        raise ValueError("an options pin lists a label twice: %r" % (value,))
    return sorted(value)


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """Name of the typed-decisions workflow whose question ids these are, else None.

    Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
    is never captured.
    """
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


def _question_schema(questions: Dict[str, Any]) -> str:
    """Order-sensitive signature of a question schema, for sharing forward passes.

    sort_keys=False keeps insertion order significant at every nesting level, because
    option order is positional in render_options. default=str matches render_criterion's
    tolerance, so schemas that render identically still share a group.
    """
    return json.dumps(questions, sort_keys=False, ensure_ascii=False, default=str)


# Subtags that mean "the English checkpoint can read this". Routing needs one bit -- is this
# English Latin text, or something the English checkpoint cannot read -- not a language id, so
# every other code that names a language resolves to the multilingual checkpoint.
_ENGLISH_SUBTAGS = ("en", "eng", "english")

# Codes that are valid `$LANG` values but name no language, so they answer nothing about the
# state. `C`, `POSIX` and `C.UTF-8` are what minimal images ship -- `C.UTF-8` is the default
# `LANG` in the official Python image, which is where `laya-serve` runs -- and the ISO 639-2
# special codes say the same thing in the standard's own vocabulary: `und` undetermined,
# `zxx` no linguistic content, `mul` multiple languages. They abstain, which is what the blank
# case below already does, rather than forcing the multilingual checkpoint on English text.
_LANGUAGE_AGNOSTIC_CODES = ("c", "posix", "und", "zxx", "mul")


def _english_from_code(value: Any) -> Optional[bool]:
    """True/False for a language code, or None when the code identifies nothing.

    Accepts the forms a caller is likely to have to hand: `"en"`, `"EN"`, `"en-US"`, the
    POSIX `"en_US"` (which `$LANG` holds), and `"en_US.UTF-8"`. `None` here means "no usable
    hint", which is what lets a language-identification model abstain -- and it is also what a
    code that names no language returns, so `LANG=C` falls through to detection instead of
    pinning every request to one checkpoint.
    """
    if value is None:
        return None
    code = str(value).strip().lower()
    if not code:
        return None
    code = code.split(".", 1)[0]                       # en_US.UTF-8 -> en_US
    primary = code.replace("_", "-").split("-", 1)[0]  # en_US -> en
    if not primary or primary in _LANGUAGE_AGNOSTIC_CODES:
        return None
    return primary in _ENGLISH_SUBTAGS


class _ScanLong:
    """Last start hook of `Router.predict_long`: score the routed state with the agent's
    `predict_long` rather than leave `predict` to call `system_one` on one window of it.

    It is appended after every other start hook, so a caller's hook has already had its say: one
    that answered (`ctx.skip(...)`) or rewrote the state/questions wins, and only what is left is
    scanned. The `lang` rule is `predict`'s -- an explicit `lang=`, else the language routing
    detected -- repeated here because `predict` computes it locally for its own call.
    """

    def __init__(self, window, stride, aggregate, batch_size, lang):
        self.window = window
        self.stride = stride
        self.aggregate = aggregate
        self.batch_size = batch_size
        self.lang = lang

    def on_predict_start(self, ctx):
        if ctx.results is not None:
            return
        if not hasattr(ctx.agent, "predict_long"):
            raise TypeError(
                "%s has no predict_long, so a state longer than its window cannot be scanned; "
                "the PyTorch Agent implements it, and an ONNX or hand-attached agent needs it too"
                % type(ctx.agent).__name__
            )
        lang = self.lang
        if lang is None:
            lang = (ctx.decision.get("detection") or {}).get("language")
        ctx.results = [ctx.agent.predict_long(
            ctx.states[0], ctx.questions, window=self.window, stride=self.stride,
            aggregate=self.aggregate, batch_size=self.batch_size, lang=lang)]


# Checkpoint options a Router will not accept through `agent_kwargs`: the ones it sets for itself
# on every `Agent(...)` it builds, plus the hook family, which has its own home in `Router(hooks=)`
# and would otherwise fire from two registries at once.
_ROUTER_OWNED_AGENT_ARGS = frozenset({
    "model_id_or_path", "device", "token", "subfolder", "revision",
    "hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_concurrent",
    "hooks_timeout",
})


def check_agent_kwargs(agent_kwargs: Dict[str, Any]) -> None:
    """Reject `agent_kwargs` names a Router cannot honour, before any checkpoint is loaded.

    The accepted names are read out of `Agent.__init__`'s signature rather than written down here:
    the whole point of `agent_kwargs` is that this file stops keeping its own list of checkpoint
    options, so a hard-coded copy of them would go stale in exactly the way this argument removes.
    The import is only reached when a caller passes the argument, which is also the moment torch is
    about to be loaded anyway.
    """
    from .agent import Agent

    accepted = set(inspect.signature(Agent.__init__).parameters) - {"self"}
    reserved = sorted(set(agent_kwargs) & _ROUTER_OWNED_AGENT_ARGS)
    if reserved:
        raise ValueError(
            "Router sets %s itself; pass them to Router(...) instead of in agent_kwargs"
            % (", ".join(repr(n) for n in reserved),))
    unknown = sorted(set(agent_kwargs) - accepted)
    if unknown:
        raise ValueError(
            "Agent accepts none of %s; it accepts %s"
            % (", ".join(repr(n) for n in unknown), sorted(accepted - _ROUTER_OWNED_AGENT_ARGS)))


def _digests_from_env(models: Dict[str, Any], resolve=normalise_name) -> Dict[str, Optional[Dict[str, str]]]:
    """Turn `LAYA_SHA256_DIGESTS` into per-checkpoint digest maps when it names models.

    The variable has two shapes, and the value types say which. A flat
    `{artifact: digest}` map is `laya.revisions.verify_digests`'s own reading -- the same
    files checked on every checkpoint the process loads -- and is returned as `{}` here so
    that path stays untouched. A nested `{model: {artifact: digest}}` map pins each
    checkpoint with its own files, which is what a server holding several resident needs:
    the bundled repository ships a separate `model.safetensors` per checkpoint, so one flat
    map can only ever match one of them and refuses the rest at startup.

    A checkpoint the nested map does not name is returned as `{}`, meaning "not covered by this
    variable, and do not let `verify_digests` read this nested map as an artifact map either" -- it
    would fail with `cannot verify 'english': no such file` on a checkpoint the variable never
    named. It needs no distinguishing from a `{}` somebody wrote: both name no files, and
    `_merge_expected_digests` merges them identically.
    Keys are normalised exactly as `Router(sha256_digests=...)` normalises them, so `en`
    names `english` and a name core does not know raises here rather than quietly leaving
    that checkpoint unverified. Unset, empty or unparseable input names nothing:
    `laya.revisions` reports a malformed value in its own words.
    """
    raw = os.environ.get("LAYA_SHA256_DIGESTS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict) or not data:
        return {}
    values = list(data.values())
    if all(isinstance(v, str) for v in values):
        return {}                                     # flat: laya.revisions already applies it
    if not all(isinstance(v, dict) for v in values):
        raise ValueError("LAYA_SHA256_DIGESTS must be either {artifact: digest} for every "
                         "checkpoint or {model: {artifact: digest}} per checkpoint; %s mixes "
                         "the two or holds a value that is neither" % sorted(data))
    per_model = {resolve(k): dict(v) for k, v in data.items()}
    for name in models:
        per_model.setdefault(resolve(name), {})
    return per_model


def _revision_pin(value: Optional[str]) -> Optional[str]:
    """The revision `value` asks for, or None when it asks for nothing.

    Unset and blank are the same answer, because both are what a configuration line that did not
    get filled in leaves behind, and neither is a commit anything can be pinned to.
    """
    return value if value is not None and str(value).strip() else None


def _digest_entry(
    digests: Dict[str, Optional[Dict[str, str]]],
    key: str,
) -> Optional[Dict[str, str]]:
    """One checkpoint's own digest entry, as `Router.sha256_digests` holds it right now.

    None when the checkpoint has no entry at all, which is what leaves `verify_digests` its own
    environment fallback; `{}` for an entry that names no files, whether that is the placeholder a
    nested `LAYA_SHA256_DIGESTS` left or a `None` a caller wrote.

    Read here rather than decided once at construction, because `sha256_digests` is a public mutable
    attribute: a per-checkpoint pin assigned to it -- or added to an existing entry in place -- after
    `Router(...)` has to count, and deciding from a key set frozen at construction dropped exactly
    that.
    """
    if key not in digests:
        return None
    entry = digests[key]
    return {} if entry is None else entry


def _merge_expected_digests(
    shared: Optional[Dict[str, str]],
    per_model: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """The `expected_sha256` one checkpoint is built with, out of the two channels that pin it.

    `shared` is `agent_kwargs["expected_sha256"]`, which the class docstring advertises and which
    reaches every checkpoint the Router builds. `per_model` is this checkpoint's own entry: None when
    it has none at all, and `{}` for the placeholder a nested `LAYA_SHA256_DIGESTS` leaves for a
    checkpoint it does not name.

    None -- both channels silent -- means "pass no `expected_sha256`", which leaves `verify_digests`
    its own `LAYA_SHA256_DIGESTS` fallback and keeps an unconfigured load byte for byte what it was.
    An empty `{}` is **not** the same thing and must still reach `Agent`: it masks that fallback,
    which for a *nested* variable would otherwise be read as an artifact map and raise
    `cannot verify 'english': no such file` on a checkpoint the variable never named.

    A value that is not a mapping is handed on untouched, so `verify_digests` keeps ownership of the
    "expected_sha256 must be a mapping" message rather than this layer growing a second copy.

    The two channels are merged file by file, because each names files and a digest is a claim about
    one file -- dropping either half would verify less than the caller asked for. **For a file both
    name, the per-checkpoint entry wins.** They are not equally specific: `shared` reaches every
    checkpoint, and `model.safetensors` is the one name every checkpoint uses for a *different* file,
    so a shared entry for it cannot be a correct claim about all of them at once. Refusing that
    overlap as a contradiction broke the ordinary shape it appears in -- a shared pin plus a
    per-checkpoint override, which loaded correctly before this function existed.

    An empty placeholder therefore needs no special case: merging it changes nothing.
    """
    if per_model is not None and not isinstance(per_model, dict):
        return per_model
    if shared is not None and not isinstance(shared, dict):
        return shared
    if shared is None and per_model is None:
        return None
    merged = dict(shared or {})
    merged.update(per_model or {})
    return merged


class _InFlightBuild:
    """Private synchronization descriptor for an in-flight checkpoint build.

    `done` is signalled once the build finishes (or fails), releasing callers waiting
    for this specific checkpoint. `error` records any exception raised during the build
    so concurrent waiting callers unblock and receive the failure instead of deadlocking.
    """
    __slots__ = ("done", "error")

    def __init__(self):
        self.done = threading.Event()
        self.error: Optional[BaseException] = None


def _pin(key: str, qid: Any, value: Any) -> str:
    """The digest a pin of `value` on question `qid` stands for; refuses what could never match."""
    if not isinstance(value, (str, MappingABC, list)) or (isinstance(value, str) and not value.strip()):
        raise TypeError("signature for %r pins question %r to %r; expected the instructions text, "
                        "its sha1: digest, or null" % (key, qid, value))
    return instructions_digest(value)


def parse_signature(key: str, raw: Any) -> Signature:
    """A `Signature` from any accepted form: an id list, an id -> instructions mapping, the faceted
    object `registered` reports, or a `Signature`. Refuses what could never match. `key` names the
    checkpoint in error messages. Module-level so the store validates manifests with the same code
    and without the Router class, which tests may replace.
    """
    options: Dict[str, Any] = {}
    purpose: Optional[str] = None
    if isinstance(raw, Signature):      # same checks as the other forms: text pins hash, digests normalise
        ids = raw.ids
        if not all(isinstance(q, str) for q in ids):
            raise TypeError("signature for %r has non-text question ids %r" % (key, sorted(ids, key=repr)))
        pins = {q: _pin(key, q, v) for q, v in raw.pins.items() if v is not None}
        options, purpose = dict(raw.options), raw.purpose
    elif _is_described(raw):    # the faceted shape `registered` reports, so it round-trips
        if not all(isinstance(q, str) for q in raw["ids"]):
            raise TypeError("signature for %r: 'ids' must be a list of question-id strings, got %r"
                            % (key, raw["ids"]))
        ids = frozenset(raw["ids"])
        pins = {}
        for qid, value in raw.get("instructions", {}).items():
            if value is not None:
                pins[str(qid)] = _pin(key, qid, value)
        for qid, value in raw.get("options", {}).items():
            if value is not None:
                options[str(qid)] = value
        purpose = raw.get("for")
    elif isinstance(raw, MappingABC):
        if "ids" in raw:    # the faceted shape, malformed; a question id literally named "ids" is not supported
            raise TypeError("signature for %r has an 'ids' key but is not a valid faceted object: it takes "
                            "'ids' (a list) and optionally 'instructions' and 'options' (mappings) and 'for' "
                            "(text), and nothing else, got %r" % (key, raw))
        ids = frozenset(str(q) for q in raw)
        pins = {}
        for qid, value in raw.items():
            if value is None:
                continue
            pins[str(qid)] = _pin(key, qid, value)
    elif isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise TypeError("each signature for %r must be a set of question ids, a mapping from question id "
                        "to instructions, or a faceted {ids, instructions, options, for} object, got %r"
                        % (key, raw))
    else:
        ids, pins = frozenset(str(q) for q in raw), {}
    if purpose is not None and (not isinstance(purpose, str) or not purpose.strip()):
        raise TypeError("signature for %r: 'for' must be a non-empty string, got %r" % (key, purpose))
    if not ids or any(not q.strip() for q in ids):
        raise ValueError("a signature for %r is empty or holds a blank question id: %r" % (key, sorted(ids)))
    for facet, pinned in (("instructions", set(pins)), ("options", set(options))):
        if not pinned <= ids:
            raise ValueError("signature for %r pins %s of %s, which are not among its ids %s"
                             % (key, facet, sorted(pinned - ids), sorted(ids)))
    return Signature(ids, pins, options, purpose)


class Router(HookRegistry):
    """Lazily loads Laya checkpoints and sends each request to the right one.

        from laya import Router

        r = Router()
        r.predict({"message": "Mein Konto wurde zweimal belastet"}, questions)   # -> multilingual
        r.predict({"message": "I was charged twice"}, questions)                 # -> english
        r.predict(state, questions, model="typed-decisions")                     # explicit

    Models are downloaded and built on first use. `max_loaded` caps how many stay resident
    (least-recently-used is evicted), because all three together are ~1.16B parameters.

    The default is 2, because automatic routing only ever chooses between `english` and
    `multilingual`: a cap of one rebuilds the checkpoint it just evicted on every script switch,
    which is seconds per request on exactly the traffic the Router exists for. Traffic that only
    ever sees one language never builds the second checkpoint, so the default costs it nothing.
    Lower it to 1 for a memory-constrained host, and raise it to 3 (or preload) when
    `auto_task_detection`, an explicit `model=` or an explicit `task=` can reach
    `typed-decisions` as well.

    For a server or a demo, preload instead: a cold load costs seconds, while detection costs
    microseconds, so even the default still pays a load the first time a language appears.

        r = Router(preload=True)                    # all three resident, routing is free
        r = Router(preload=True, device="cuda")
        r.preload(["english", "multilingual"])      # or just the two you serve

    Hub revisions are opt-in. `revision` applies one commit to every model;
    `revisions={"english": "...", "multilingual": "..."}` overrides that per model,
    which is useful when standalone repositories were reviewed at different commits.
    Without either, huggingface_hub's normal default and existing offline cache are used.

    A `revisions` entry that is `None` or blank is "no override for this model", so the model
    inherits `revision` -- the shape `{"english": os.environ.get("EN_SHA")}` writes when the
    variable is unset, which must not cost the caller the pin it did ask for. Nothing in
    `revisions` can unpin one model while `revision` pins the rest; leave `revision` unset and
    name the models you want pinned instead.

    A blank `revision` is read the same way, which is a deliberate change in what the Router
    passes on: `Router(revision="   ")` used to reach `resolve_revision`, where a truthy but blank
    string suppressed the `LAYA_REVISION` fallback and left huggingface_hub's default, and is now
    dropped before it gets there, so a Router configured with whitespace behaves like one
    configured with nothing and `$LAYA_REVISION` applies. That is what "this configuration line was
    never filled in" has to mean if `revision` and a `revisions` entry are to be read the same way.

    It is one of two places a weaker source ends up ahead of an explicit argument. The other is a
    per-checkpoint digest entry from `LAYA_SHA256_DIGESTS`, which wins over an `expected_sha256`
    passed through `agent_kwargs`; see the class docstring.

    Anything else `laya.Agent` accepts is reachable through `agent_kwargs`, which is merged into
    every checkpoint the Router builds:

        Router(agent_kwargs={"lang_temperatures": {"de": {"temperature": [1.0, 1.4, 2.0]}}})
        Router(agent_kwargs={"expected_sha256": {"model.safetensors": "a3f1..."}})
        Router(agent_kwargs={"fast": True})

    `expected_sha256` there pins the same files on every checkpoint, which is what a single
    resident checkpoint or a shared `tokenizer.json` wants. It is never discarded wholesale by a
    checkpoint's own digests: where one also has an entry, from `sha256_digests` or from a
    **per-checkpoint** `LAYA_SHA256_DIGESTS`, the two maps are merged **file by file**, so a file
    only one of them names is still verified.

    Two exceptions, both deliberate and both tested, because "merged file by file" is not the whole
    story and the difference is a supply-chain control:

    * A **flat** `LAYA_SHA256_DIGESTS` -- `{artifact: digest}` rather than `{model: {...}}` -- is
      not a layer here at all. `verify_digests` applies it itself, but only when nothing else pins
      (`if expected is None`), so ANY `expected_sha256` reaching `Agent`, from here or from a
      checkpoint entry, means the flat variable is not consulted for that load. Verified on real
      files: a flat variable pinning `model.safetensors` plus an `agent_kwargs` map pinning
      `tokenizer.json` loads a tampered `model.safetensors`. Use the per-checkpoint shape, or name
      every file you care about in one map, if you need both. This is unchanged from `main`.
    * An explicit `{}` or `None` entry MASKS what would otherwise apply -- that is what "load this
      one unverified" has to mean, and `test_an_explicit_none_entry_masks_a_flat_environment_map`
      pins it.

    And the precedence is per-checkpoint over shared regardless of where each came from, so a
    per-checkpoint entry synthesised from `LAYA_SHA256_DIGESTS` wins over an `expected_sha256`
    passed here in code. An environment variable beating an explicit argument is worth stating
    plainly on a control like this; `test_an_environment_pin_overrides_the_shared_one_per_checkpoint`
    is where that is pinned.

    For a file both name, the per-checkpoint entry wins. The two are not equally specific: the
    `agent_kwargs` map reaches every checkpoint the Router builds, and `model.safetensors` is the one
    name every checkpoint uses for a *different* file, so a shared entry for it cannot be a correct
    claim about all of them at once. **Nothing raises over that overlap** -- refusing it would reject
    a shared pin plus a per-checkpoint override, which is the ordinary shape and which loaded
    correctly before any of this existed. If you need to know which digest a checkpoint was verified
    against, read it back: the map handed to each `Agent` is the merge described above.

    Both `agent_kwargs` and `sha256_digests` are public and mutable, and a checkpoint's entry is read
    on the load rather than at construction, so a pin assigned afterwards -- or added to an
    existing entry in place -- counts.

    The names the Router sets for itself -- `model_id_or_path`, `device`, `token`, `subfolder`,
    `revision` and the hook arguments -- are refused here rather than silently shadowed, and the
    remaining names are checked against `Agent.__init__` at construction, so a misspelled option
    fails on the `Router(...)` line instead of on the first request.

    Artifact digests are opt-in and always per model: `sha256_digests={"english": {...}}`
    passes that `{path relative to the checkpoint dir: hexdigest}` map to the `Agent` that
    loads it, so a tampered or substituted weight file is refused before it is parsed. There
    is no Router-wide equivalent of `revision` because digests, unlike a commit SHA, are not
    shareable: the bundled repository ships a separate `model.safetensors` for each of
    `english`, `multilingual` and `typed-decisions`, so one flat map can only ever match one
    of them. A model listed with `None` or `{}` adds no files of its own, which loads it
    unverified unless `agent_kwargs["expected_sha256"]` pins it; it still masks a flat
    `LAYA_SHA256_DIGESTS`, which is what listing it that way is for.

    The same split is available to a process configured only by environment: when
    `LAYA_SHA256_DIGESTS` holds a model-keyed map (`{"english": {...}, "multilingual": {...}}`)
    this seeds it per checkpoint, so a server that keeps several resident can pin each with its
    own digests instead of refusing to start on the second one. A flat `LAYA_SHA256_DIGESTS`
    keeps its existing meaning, applied by `laya.revisions` to every checkpoint the process
    loads, which is right for a single-checkpoint one. An argument entry wins over the
    environment for the model it names. A nested variable that names some checkpoints and not
    others says nothing about the others: they keep whatever `agent_kwargs["expected_sha256"]`
    pins them with, because pinning one checkpoint from the environment is not a request to stop
    verifying the rest.

    Hooks are opt-in and run at the Router level: `on_route` sees the routing decision,
    `on_load` / `on_evict` see model lifecycle, and `on_predict_start` / `on_predict_end`
    wrap the whole route+infer call. See `laya.hooks`.
    """

    # Opt-in defaults so a hand-built instance (`Router.__new__` in tests) works unset.
    # `hooks`/`_hooks_mutex` come from HookRegistry.
    hooks_raise = True
    hooks_concurrent = True
    hooks_timeout = None
    _hooks_lock = None
    agent_kwargs: Dict[str, Any] = {}

    def __init__(
        self,
        models: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        revisions: Optional[Dict[str, Optional[str]]] = None,
        max_loaded: int = 2,
        default: str = "english",
        auto_task_detection: bool = False,
        standalone_repos: bool = False,
        preload: bool = False,
        lang_guess: Optional[Any] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: bool = True,
        hooks_concurrent: bool = True,
        hooks_timeout: Optional[float] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        sha256_digests: Optional[Dict[str, Optional[Dict[str, str]]]] = None,
        signatures: Optional[Dict[str, Iterable[Any]]] = None,
    ):
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self._hooks_mutex = threading.Lock()
        # The registry: built-in names first, then whatever the caller adds. A key that is a
        # built-in name re-points that checkpoint (a fine-tune standing in for `english`); any
        # other key registers a checkpoint of the caller's own, served beside the built-ins and
        # named in `model=` like one of them. `signatures` maps a registered name to the question-id
        # sets that select it without `model=`, the way the typed-decisions workflows select theirs.
        self.models: Dict[str, Any] = dict(STANDALONE_MODELS if standalone_repos else DEFAULT_MODELS)
        self.signatures: Dict[str, List[Signature]] = {}
        # Free text per registered checkpoint, reported by `registered` (and so by `/health` and
        # `laya_status`); a public mutable attribute like `sha256_digests`, filled by `register`.
        self.descriptions: Dict[str, str] = {}
        for k, v in (models or {}).items():
            self._add(k, v)
        for k, sigs in (signatures or {}).items():
            self._add_signatures(self.resolve(k), sigs)
        self.device = device
        self.token = token or os.environ.get("HF_TOKEN")
        # Optional Hub revision (commit SHA/branch/tag) applied to every checkpoint load.
        # `revisions` overrides it per normalized model name, for standalone repos whose
        # reviewed commits differ.
        self.revision = revision
        self.revisions: Dict[str, Optional[str]] = {
            self.resolve(k): v for k, v in (revisions or {}).items()
        }
        # Options forwarded to every `Agent` this Router builds, checked now so a bad name fails on
        # this line rather than the first request that happens to load a checkpoint.
        self.agent_kwargs: Dict[str, Any] = dict(agent_kwargs or {})
        if self.agent_kwargs:
            check_agent_kwargs(self.agent_kwargs)
        # Per checkpoint SHA-256 map: seeded from a model-named `LAYA_SHA256_DIGESTS`, then
        # overridden checkpoint by checkpoint by the argument. Keyed and normalised exactly like
        # `revisions`, so a misspelled model name fails here rather than leaving that checkpoint
        # unverified.
        self.sha256_digests: Dict[str, Optional[Dict[str, str]]] = _digests_from_env(self.models, self.resolve)
        argument = {self.resolve(k): v for k, v in (sha256_digests or {}).items()}
        self.sha256_digests.update(argument)
        self.max_loaded = max(1, int(max_loaded))
        self.default = self.resolve(default)
        self.auto_task_detection = bool(auto_task_detection)
        # An opt-in language hint installed for every request: a code, or a callable taking the
        # state and returning one (or None to abstain). Checked before the built-in detection,
        # never before an explicit `model`, `task` or `lang`. The default path is unchanged, so
        # the heuristic stays dependency-free; this is the seam for a real LID model.
        self.lang_guess = lang_guess
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []          # least-recently-used first
        # Synchronization invariants:
        # - `_lock`: Re-entrant lock guarding Router shared state: `_agents`, `_order`,
        #   the `_loading` registry, and lifecycle bookkeeping. Never held during expensive
        #   checkpoint construction (Agent download/init), nor while waiting on in-flight events.
        # - `_build_lock`: Non-reentrant lock serializing expensive Agent construction globally
        #   (one build at a time across all checkpoints to preserve peak memory bounds).
        #   Acquired outside `_lock` (never inside) to prevent deadlocks, and released before
        #   dispatching lifecycle hooks.
        # - `_loading[name]`: Private per-checkpoint in-flight build registry. Maps normalized
        #   model names to their `_InFlightBuild` descriptor so concurrent loads deduplicate (#95)
        #   and `unload(name)` waits only for builds of the SAME checkpoint without stalling
        #   on unrelated builds holding `_build_lock`.
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._loading: Dict[str, _InFlightBuild] = {}
        if preload:
            self.preload()

    # ------------------------------------------------------------------ registry
    def resolve(self, name: Any) -> str:
        """Canonical key of a checkpoint this Router knows: built-in, aliased, or registered.

        The instance counterpart of :func:`normalise_name`. Everything on the Router that takes a
        checkpoint name goes through here, so a checkpoint registered under `papers` is accepted
        wherever `english` is: `model=`, `task=`, `load`, `unload`, `preload`, `revisions`.
        """
        key = canonical_name(name)
        if key in self.models:
            return key
        raise ValueError("unknown model %r; choose one of %s (or an alias: %s)"
                         % (name, sorted(self.models), sorted(_ALIASES)))

    def register(self, name: str, source: Any, signatures: Optional[Iterable[Any]] = None,
                 description: Optional[str] = None) -> str:
        """Add a checkpoint under `name`, to be served beside the built-in ones.

        `source` is what `laya.load` accepts: a Hub repo id, a `(repo, subfolder)` pair, or a local
        directory holding `model.safetensors` and `rl_agent_config.json` -- a fine-tune exported by
        the training recipe, for instance. Nothing is downloaded or built here; the checkpoint loads
        on first use like the built-ins, is evicted and unloaded like them, and counts toward
        `max_loaded` like them.

        `signatures` lists the ways a request selects this checkpoint when it names no `model`.
        Each is either a set of question ids (`["section"]`): a request whose question ids are
        exactly that set goes here, the same exact match the typed-decisions workflows use; or a
        mapping from question id to the instructions text of that question (or its
        `instructions_digest`), which also requires the request's question under that id to carry
        those instructions. The mapping form ties the checkpoint to the questions it was trained on,
        so two workflows that both ask a `section` question with different wording are told apart.
        The faceted object `{"ids": [...], "instructions": {...}, "options": {...}, "for": "..."}` also
        pins the answer set of a question (`options`: sorted choice labels or a score level count);
        every facet present must match, and `for` is documentation only.
        Several signatures may name one checkpoint, because a fine-tune usually covers a few related
        question formats. A signature already registered for another checkpoint is refused, since
        the router could not tell the two apart. Passing `signatures` for a name that is already
        registered replaces its signatures (the list given is the whole set); leaving it out keeps them.

        `description` is free text about the checkpoint for whoever reads `registered`.

        Returns the canonical name. Registering an existing name replaces its source.
        """
        with self._lock:
            key = canonical_name(name)
            missing = object()      # a present None (an opted-out digest entry) is not an absent key
            before = (self.models.get(key, missing), self.sha256_digests.get(key, missing),
                      self.descriptions.get(key, missing),
                      list(self.signatures[key]) if key in self.signatures else missing)
            try:
                key = self._add(name, source)
                if key not in self.sha256_digests:
                    # A nested LAYA_SHA256_DIGESTS names no file of this checkpoint: the `{}` placeholder
                    # the constructor gives every model keeps `verify_digests` off the nested map.
                    placeholder = _digests_from_env([key], canonical_name).get(key)
                    if placeholder is not None:
                        self.sha256_digests[key] = placeholder
                if signatures is not None:
                    self.signatures.pop(key, None)      # the list given is the whole set, not an addition
                    self._add_signatures(key, signatures)
                if description is not None:
                    self.descriptions[key] = str(description)
            except BaseException:       # a refused register() leaves the router as it was
                for table, old in ((self.models, before[0]), (self.sha256_digests, before[1]),
                                   (self.descriptions, before[2]), (self.signatures, before[3])):
                    if old is missing:
                        table.pop(key, None)
                    else:
                        table[key] = old
                raise
        return key

    def unregister(self, name: str) -> None:
        """Remove a registered checkpoint: its source, signatures, description, pins and resident Agent.

        A built-in name is refused (re-point it with `register`, it cannot be removed), and so is a
        name the router does not know. A resident Agent is unloaded first, waiting for an in-flight
        build of it, so a request that already holds it finishes; a request that names it afterwards
        gets the usual unknown-model error.
        """
        key = self.resolve(name)
        if key in DEFAULT_MODELS:
            raise ValueError("%r is a built-in checkpoint and cannot be unregistered" % key)
        with self._lock:
            self.models.pop(key, None)      # first, so no new load can start a build of it
            self.signatures.pop(key, None)  # and no request can match it into the missing model
        freed = self._unload_key(key)       # waits for a build that had already read its source
        with self._lock:
            for table in (self.signatures, self.descriptions, self.sha256_digests, self.revisions):
                table.pop(key, None)
            self._agents.pop(key, None)
            if key in self._order:
                self._order.remove(key)
        self._dispatch_lifecycle("on_evict", freed)

    @property
    def registered(self) -> Dict[str, Dict[str, Any]]:
        """The checkpoints registered beside the built-ins: name -> source, signatures, description.

        Serialisable as it is; `/health` and `laya_status` report it under `registered`.
        """
        with self._lock:
            return {
                name: {
                    "source": _repo_str(spec),
                    "signatures": [sig.describe() for sig in self.signatures.get(name, [])],
                    "description": self.descriptions.get(name),
                }
                for name, spec in self.models.items() if name not in DEFAULT_MODELS
            }

    def _add(self, name: Any, source: Any) -> str:
        """`register` without the lock, for the constructor."""
        key = canonical_name(name)
        if key not in DEFAULT_MODELS:
            if not isinstance(name, str) or not _NAME_RE.match(key):
                raise ValueError("invalid checkpoint name %r: use lowercase letters, digits, '.', '_' or '-', "
                                 "starting with a letter or digit" % (name,))
            if key == "auto":
                raise ValueError("invalid checkpoint name %r: 'auto' means automatic routing in every "
                                 "model= argument" % (name,))
        if source is not None and not isinstance(source, (str, tuple, list)):
            raise TypeError("checkpoint source for %r must be a repo id, a local path or a (repo, subfolder) "
                            "pair, got %s" % (name, type(source).__name__))
        if isinstance(source, str) and source.startswith("~"):
            source = os.path.expanduser(source)     # a local path, never a Hub repo id
        self.models[key] = source
        return key

    def _add_signatures(self, key: str, signatures: Iterable[Any]) -> None:
        if isinstance(signatures, (str, bytes, MappingABC)) or not isinstance(signatures, Iterable):
            raise TypeError("signatures for %r must be a list of question-id sets or id-to-instructions mappings" % key)
        for raw in signatures:
            sig = self._parse_signature(key, raw)
            for other, sigs in self.signatures.items():
                if other != key and any(sig.conflicts(prior) for prior in sigs):
                    raise ValueError("signature %s is already registered for %r or cannot be told apart from one "
                                     "of its signatures; a request must select one checkpoint only"
                                     % (sig.describe(), other))
            self.signatures.setdefault(key, [])
            if sig not in self.signatures[key]:
                self.signatures[key].append(sig)

    _pin = staticmethod(lambda key, qid, value: _pin(key, qid, value))

    @staticmethod
    def _parse_signature(key: str, raw: Any) -> Signature:
        return parse_signature(key, raw)

    def match_signature(self, questions: Optional[Dict[str, Any]],
                        workflow: Any = _NOT_COMPUTED) -> Optional[Tuple[str, str]]:
        """`(model, label)` for the checkpoint whose registered signature these question ids are, else None.

        Registered signatures are checked first and always: registering one is the opt-in. The four
        built-in typed-decisions workflows come after, and only with `auto_task_detection=True`,
        as before. The label is the workflow name for a built-in match and the checkpoint name for
        a registered one; it lands in `RouteDecision["workflow"]`. A caller that already ran
        `match_typed_decisions_workflow` on these questions passes the result as `workflow`, None
        included (no match), so the workflow sets are walked once.
        """
        if questions:
            with self._lock:    # `register` may be adding signatures from another thread
                registered = list(self.signatures.items())
            # Most pinned facets first: one that also names the instructions or the options is the
            # more specific claim, so two checkpoints may share an id set and be told apart.
            candidates = [(model, sig) for model, sigs in registered for sig in sigs]
            candidates.sort(key=lambda c: -c[1].specificity())  # stable: most pinned facets first
            for model, sig in candidates:
                if sig.matches(questions):
                    return model, model
        if workflow is _NOT_COMPUTED:
            workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return "typed-decisions", workflow
        return None

    # ------------------------------------------------------------------ loading
    def load(self, name: str):
        """Return the Agent for `name`, downloading and building it on first use.

        Concurrent callers share a single Agent instead of building duplicates.
        """
        key = self.resolve(name)
        while True:
            with self._lock:
                if key in self._agents:
                    self._touch(key)
                    return self._agents[key]
                inflight = self._loading.get(key)
                if inflight is None:
                    inflight = _InFlightBuild()
                    self._loading[key] = inflight
                    break
            inflight.done.wait()
            if inflight.error is not None:
                raise inflight.error

        agent = None
        evicted = []
        try:
            with self._build_lock:
                with self._lock:
                    # Built (or attached) while this caller waited for the build lock.
                    if key in self._agents:
                        self._touch(key)
                        agent = self._agents[key]
                        if self._loading.get(key) is inflight:
                            self._loading.pop(key, None)
                        inflight.done.set()
                        return agent
                built_agent = self._build(key)
                with self._lock:
                    if key in self._agents:      # attached while it was building: keep that one
                        self._touch(key)
                        agent = self._agents[key]
                    else:
                        agent = built_agent
                        self._agents[key] = agent
                        self._order.append(key)
                        evicted = self._evict_locked()
                    if self._loading.get(key) is inflight:
                        self._loading.pop(key, None)
                    inflight.done.set()
        except BaseException as e:
            with self._lock:
                if self._loading.get(key) is inflight:
                    self._loading.pop(key, None)
                inflight.error = e
                inflight.done.set()
            raise

        # Lifecycle hooks fire after the lock is released, so a hook can safely call the Router.
        self._dispatch_lifecycle("on_evict", evicted)
        dispatch(compose_hooks(self.hooks), "on_load",
                 PredictContext(states=[], questions={}, model=key, agent=agent, router=self),
                 raise_errors=self.hooks_raise, lock=self._hooks_lock, timeout=self.hooks_timeout)
        return agent

    def _build(self, key: str):
        """Construct the Agent for `key`: config is read under `_lock`, the build runs outside it."""
        from .agent import Agent
        with self._lock:
            if self.models.get(key) is None:
                raise ValueError("checkpoint %r has no source to load from: it was attached as a built Agent "
                                 "and is no longer resident; attach it again, or register a path or repo" % key)
            repo, sub = _split(self.models[key])
            kwargs = {"device": self.device, "token": self.token, "subfolder": sub}
            # A None or blank entry in `revisions` is "no pin of its own", so the checkpoint
            # inherits `revision`. Reading it as a pin of nothing dropped the caller's `revision`
            # and handed that one checkpoint to `$LAYA_REVISION` instead, which is the opposite of
            # what asking for a pin means.
            model_revision = _revision_pin(self.revisions.get(key)) or _revision_pin(self.revision)
            if model_revision is not None:
                kwargs["revision"] = model_revision
            # Safe to merge last for the names above, which `agent_kwargs` refuses -- but not for
            # `expected_sha256`, which is deliberately reachable through `agent_kwargs` and is
            # therefore merged with this checkpoint's entry rather than replaced by it.
            kwargs.update(self.agent_kwargs)
            expected = _merge_expected_digests(kwargs.pop("expected_sha256", None),
                                               _digest_entry(self.sha256_digests, key))
            if expected is not None:
                kwargs["expected_sha256"] = expected
        return Agent(repo, **kwargs)

    def _touch(self, key: str):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict_locked(self) -> List[str]:
        """Drop least-recently-used agents until `max_loaded` holds. Returns evicted names."""
        evicted: List[str] = []
        while len(self._order) > self.max_loaded:
            victim = self._order.pop(0)
            agent = self._agents.pop(victim, None)
            if agent is not None:
                evicted.append(victim)
                del agent
        if len(self._order) < len(self._agents):     # keep the two views consistent
            for k in list(self._agents):
                if k not in self._order:
                    agent = self._agents.pop(k, None)
                    if agent is not None:
                        evicted.append(k)
                        del agent
        if evicted:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        return evicted

    def _evict(self) -> List[str]:
        with self._lock:
            return self._evict_locked()

    def _dispatch_lifecycle(self, event: str, names: List[str]) -> None:
        for name in names:
            dispatch(compose_hooks(self.hooks), event,
                     PredictContext(states=[], questions={}, model=name, router=self),
                     raise_errors=self.hooks_raise, lock=self._hooks_lock, timeout=self.hooks_timeout)

    def attach(self, name: str, agent: Any):
        """Register an already-built Agent under `name` instead of loading a second copy.

        Useful when the process has a checkpoint loaded for other reasons: a demo that already
        built `convaiinnovations/laya` can hand it to the router rather than pay for -- and hold
        in memory -- a duplicate 421M parameters.

        A name the Router does not know yet is registered on the spot, with no source: the agent
        serves under that name while resident, and loading it again after an unload needs
        `register(name, source)` first.
        """
        with self._lock:
            try:
                key = self.resolve(name)
            except ValueError:
                key = self._add(name, None)
            self._agents[key] = agent
            self._touch(key)
            self.max_loaded = max(self.max_loaded, len(self._agents))
        return agent

    def preload(self, names: Optional[List[str]] = None):
        """Download and build checkpoints up front so no request ever pays a model load.

        A cold load costs seconds; language detection costs microseconds. With every
        checkpoint resident, routing is effectively free -- which is what you want in a
        server or a demo. `max_loaded` is raised to fit both the requested checkpoints and
        all already-resident agents, so incremental preloading does not evict either.
        """
        if names is None:
            # Every checkpoint that has a source to build from; an `attach`ed agent registered
            # without one is resident already and would raise here once evicted.
            with self._lock:
                names = [n for n, source in self.models.items() if source is not None]
        names = [self.resolve(n) for n in names]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(set(names) | set(self._agents)))
        for n in names:
            with self._lock:
                already = n in self._agents    # an attached agent is already built
            if not already:
                # load() dispatches on_load outside the lock; do not hold it across the call.
                self.load(n)
        return self

    def _unload_key(self, key: str) -> List[str]:
        """`unload` for a canonical key that may already be gone from `models`; returns what it freed."""
        freed: List[str] = []
        while True:
            with self._lock:
                inflight = self._loading.get(key)
                if inflight is None:
                    agent = self._agents.pop(key, None)
                    if key in self._order:
                        self._order.remove(key)
                    freed = [key] if agent is not None else []
                    del agent
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if hasattr(torch, "xpu") and torch.xpu.is_available():
                            torch.xpu.empty_cache()
                        if hasattr(torch, "mps") and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                    break
            inflight.done.wait()
        return freed

    def unload(self, name: Optional[str] = None):
        """Free one model, or all of them."""
        # When unloading a specific checkpoint, wait only for an in-flight build of THAT
        # checkpoint so unrelated builds holding `_build_lock` do not stall this unload.
        # When unloading all checkpoints (`name is None`), wait for all in-flight builds.
        freed: List[str] = []
        if name is not None:
            freed = self._unload_key(self.resolve(name))
        else:
            while True:
                with self._lock:
                    inflights = list(self._loading.values())
                    if not inflights:
                        freed = list(self._order)
                        self._agents.clear()
                        self._order.clear()
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if hasattr(torch, "xpu") and torch.xpu.is_available():
                                torch.xpu.empty_cache()
                            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                        except Exception:
                            pass
                        break
                for inflight in inflights:
                    inflight.done.wait()

        self._dispatch_lifecycle("on_evict", freed)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    @property
    def loaded_revisions(self) -> Dict[str, Optional[str]]:
        """Commit SHA each resident agent was loaded from (None for local paths)."""
        with self._lock:
            return {name: getattr(agent, "revision", None) for name, agent in self._agents.items()}

    def _resolve_hint(self, hint: Any, state: Union[str, dict, list, None]) -> Optional[bool]:
        """True/False for a hint about whether the English checkpoint can read `state`.

        `hint` is either a language code or a callable taking the state. Anything the hint
        cannot answer returns None, which makes `route` fall through to detection rather than
        picking a checkpoint on no evidence.
        """
        if hint is None:
            return None
        if callable(hint):
            hint = hint(state)
        return _english_from_code(hint)

    # ------------------------------------------------------------------ routing
    def route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, then let `on_route` hooks observe or replace it.

        `ctx.decision` is the `RouteDecision`; a hook may replace it (for example to pin a
        checkpoint) and the replacement is what gets returned and used. `hooks` are per-call
        hooks, appended after any installed on the Router.
        """
        if questions is not None and not isinstance(questions, dict):
            raise TypeError("questions must be a dict of question id -> definition, got %s" % type(questions).__name__)
        decision = self._route(state, questions, model=model, task=task, lang=lang, lang_guess=lang_guess)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        active = compose_hooks(self.hooks, hooks)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        ctx = PredictContext(states=[state], questions=questions or {}, decision=decision, router=self)
        dispatch(active, "on_route", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
        return ctx.decision

    def _route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, without loading or running anything.

        Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
        explicit `lang` > `lang_guess` > detected script/language > default.

        `lang_guess` is an opt-in hint -- a language code or a callable taking the state --
        checked after an explicit `lang` and before the built-in detection. It only answers
        "can the English checkpoint read this?", so any non-English code routes to the
        multilingual checkpoint. A hint that resolves to nothing falls through to detection,
        which lets a language-identification model abstain. Pass one here, or set
        `Router(lang_guess=...)` to apply it to every request.
        """
        # `.get`, not `[]`: a RemoteRouter resolves names its own table does not hold, because the
        # server it answers from owns the registry, and such a name has no repo to report here.
        if model is not None:
            key = self.resolve(model)
            return RouteDecision(model=key, repo=_repo_str(self.models.get(key)), reason="explicit model=%r" % model,
                                 detection=None, workflow=None)

        if task is not None:
            key = self.resolve("typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task)
            return RouteDecision(model=key, repo=_repo_str(self.models.get(key)), reason="explicit task=%r" % task,
                                 detection=None, workflow=None)

        workflow = match_typed_decisions_workflow(questions or {})
        matched = self.match_signature(questions, workflow)
        matched_source = self.models.get(matched[0], _MISSING) if matched is not None else None
        if matched is not None and matched_source is _MISSING:
            matched = None      # unregistered since the match: route by language, as if nothing matched
        if matched is not None:
            key, label = matched
            if key == "typed-decisions" and label == workflow:
                reason = "question ids match the %r typed-decisions workflow" % workflow
            else:
                reason = "question ids match a signature registered for %r" % key
            return RouteDecision(model=key, repo=_repo_str(matched_source), reason=reason,
                                 detection=None, workflow=label)

        if lang is not None:
            # An explicit `lang` is decisive only when the code names a language. Blank or
            # whitespace resolves to no usable hint, so it falls through to lang_guess/detection
            # exactly as an abstaining hint does; real English/non-English codes still route now.
            resolved = _english_from_code(lang)
            if resolved is not None:
                key = "english" if resolved else "multilingual"
                return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason="explicit lang=%r" % lang,
                                     detection=None, workflow=workflow)

        # Caller-supplied hint, per-call first then the one installed on the Router. Only a hint
        # that actually answers the question routes here; anything else falls through.
        for source, hint in (("lang_guess", lang_guess), ("Router(lang_guess=...)", self.lang_guess)):
            resolved = self._resolve_hint(hint, state)
            if resolved is not None:
                key = "english" if resolved else "multilingual"
                return RouteDecision(
                    model=key, repo=_repo_str(self.models[key]),
                    reason="%s: the caller identified this as %s text" % (
                        source, "English" if resolved else "non-English"),
                    detection=None, workflow=workflow)

        det = analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it" % (
                det["script"], 100 * float(det["non_latin_fraction"]))
        elif not det["is_english"]:
            key = "multilingual"
            if det.get("mixed_segment"):
                reason = ("Latin script, mostly English, but a line or field reads as %r (%r); "
                          "the English checkpoint cannot read it" % (det["language"], det["mixed_segment"][:60]))
            elif det["language"]:
                reason = "Latin script but language looks like %r, not English" % det["language"]
            else:
                # Unidentified Latin-script language: routed on the non-English letters alone,
                # because no stopword list here covers it.
                reason = ("Latin script, language not identified but %.0f%% non-English letters; "
                          "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"])))
        elif det["language_undecided"]:
            # Nothing identifies the language: too short, or only content words ("Quero cancelar",
            # "Esqueci minha senha"). That is no evidence of English either, so it takes the same
            # `default` as a state with no letters. A deployment that serves mostly non-English
            # traffic sets `Router(default="multilingual")`; the stock default keeps it English.
            key = self.default
            reason = ("Latin script, language not identified and no non-English letters; "
                      "using default (%s)" % key)
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason=reason,
                             detection=det, workflow=workflow)

    # ------------------------------------------------------------------ running
    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Route, then answer every question in one forward pass on the chosen checkpoint.

        The result is the usual `system_one` payload plus a `routing` key recording the decision.
        Router-level `on_predict_start` / `on_predict_end` hooks wrap the whole route+infer call
        and see `ctx.decision`; see `laya.hooks`. `max_len` / `head_max_len` override the agent
        token budget for this call (a start hook may set `ctx.max_len` / `ctx.head_max_len`).
        """
        if state is None:
            raise TypeError("state must not be None; pass a string, dict, or list")
        if not isinstance(questions, dict):
            raise TypeError("questions must be a dict of question id -> definition, got %s" % type(questions).__name__)
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)

        ctx = PredictContext(
            states=[state],
            questions=questions,
            decision=None,
            model=None,
            agent=None,
            router=self,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        try:
            # Per-call hooks apply to the whole call, including on_route inside route().
            decision = self.route(state, questions, model=model, task=task, lang=lang,
                                  lang_guess=lang_guess, hooks=hooks, hooks_raise=hooks_raise,
                                  hooks_timeout=hooks_timeout)
            ctx.decision = dict(decision)
            ctx.model = decision["model"]
            agent = self.load(decision["model"])
            ctx.agent = agent
            effective_lang = lang
            if effective_lang is None and decision.get("detection") and decision["detection"].get("language"):
                effective_lang = decision["detection"]["language"]

            # Keep the existing success-path timing contract: elapsed_ms starts here, after
            # route/load and immediately before the prediction start hook.
            ctx.started_at = time.perf_counter()
            dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            if ctx.results is None:
                # Pass token-budget overrides only when set, so any Agent-like object that does
                # not accept them still works on the default path.
                overrides = {}
                if ctx.max_len is not None:
                    overrides["max_len"] = ctx.max_len
                if ctx.head_max_len is not None:
                    overrides["head_max_len"] = ctx.head_max_len

                skip = _SKIP_DEFAULTS.set(True)
                try:
                    result = agent.system_one(ctx.states[0], ctx.questions, lang=effective_lang, **overrides)
                except TypeError as e:
                    if "unexpected keyword argument 'lang'" in str(e):
                        result = agent.system_one(ctx.states[0], ctx.questions, **overrides)
                    else:
                        raise
                finally:
                    _SKIP_DEFAULTS.reset(skip)
                result["routing"] = dict(decision)
                ctx.results = [result]
            else:
                # A cache hit short-circuits inference, but Router.predict still promises a
                # `routing` key. Add it without overwriting a routing the cached payload has.
                for result in ctx.results:
                    if isinstance(result, dict):
                        result.setdefault("routing", dict(decision))
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(active, "on_error", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
                apply_confidence_gate(ctx.results, mc)
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results[0]

    def predict_long(self, state: Union[str, dict, list], questions: Dict[str, Any],
                     model: Optional[str] = None, task: Optional[str] = None,
                     lang: Optional[str] = None, lang_guess: Optional[Any] = None,
                     window: Optional[int] = None, stride: Optional[int] = None,
                     aggregate: str = "auto", batch_size: Optional[int] = None,
                     hooks=None, on_predict_start=None, on_predict_end=None,
                     hooks_raise: Optional[bool] = None, hooks_timeout: Optional[float] = None,
                     ) -> Dict[str, Any]:
        """Route, then scan every window of the state instead of only its first one.

        `predict` scores a state from a single window: anything past `max_len` is cut off (the
        first window, or for a conversation list the last) and never reaches the model. This
        routes exactly as `predict` does -- the same `model`/`task`/`lang` hints, the same
        router-level hooks, the same `routing` key and `usage` -- and scores the routed state with
        that agent's `predict_long`, which splits it into overlapping windows and aggregates per
        question. The aggregation rules are `laya.agent.Agent.predict_long`'s: `noul` takes the
        strongest window, `choice`/`score` the most confident one.

        Per-call hooks (`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`,
        `hooks_timeout`) wrap the whole route+scan exactly as they wrap `predict`: the scan runs
        last, so a start hook that answers (`ctx.skip(...)`) or rewrites the state wins.
        `max_len` / `head_max_len` are not accepted here -- a window is sized by `window` or the
        checkpoint budget, and overriding the single-window truncation is what `predict_long` is for.

        Args:
            window: state tokens per window. Defaults to the routed checkpoint's budget
                    (`max_len - head_max_len - 8`), capped at the room the questions leave for the
                    state so no window is truncated again on the way in; a smaller window isolates
                    a localized span.
            stride: token step between windows; defaults to half the effective window (50%
                    overlap). A stride past that window is a `ValueError`, since the tokens between
                    windows would reach no model.
            aggregate: "auto" (the per-type rules above) is the only mode.
            batch_size: cap on windows per forward pass, to bound memory on very long states.

        Returns:
            The usual `predict` payload, with `usage["windows"]` counting the windows scored.

        Raises:
            TypeError: the routed agent has no `predict_long` (an ONNX agent, or one attached by
                    hand), so there is nothing to scan with. Raised under the router's
                    `hooks_raise` policy, which defaults to raising.
        """
        per_call = normalise_hooks(hooks, on_predict_start, on_predict_end)
        per_call.append(_ScanLong(window=window, stride=stride, aggregate=aggregate,
                                  batch_size=batch_size, lang=lang))
        return self.predict(state, questions, model=model, task=task, lang=lang,
                            lang_guess=lang_guess, hooks=per_call,
                            hooks_raise=hooks_raise, hooks_timeout=hooks_timeout)

    def decide(self, state: Union[str, dict, list], schema: Any = None, *,
               questions: Optional[Dict[str, Any]] = None, return_details: bool = False,
               min_confidence: Optional[float] = None,
               **predict_kwargs) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        (for example `model=`, `task=`, `hooks=`) are forwarded to `predict`.
        """
        from .structured import decide as _decide
        return _decide(self, state, schema, questions=questions,
                       return_details=return_details, min_confidence=min_confidence, **predict_kwargs)

    def decide_batch(self, states: Sequence[Any], schema: Any = None, *,
                     questions: Optional[Dict[str, Any]] = None,
                     return_details: bool = False, min_confidence: Optional[float] = None,
                     **predict_kwargs) -> List[Any]:
        """Answer many states against one schema (JSON schema or pydantic model) in one batched call.

        The throughput form of :meth:`decide`: the schema is planned once and its questions
        run over every state through :meth:`predict_batch` (grouped forward passes, results in
        input order), then each state's answers are projected as ``decide`` does. Extra keyword
        arguments (``batch_size=``, ``model=``, ``hooks=``, ...) are forwarded to
        ``predict_batch``. See `laya.structured`.
        """
        from .structured import decide_batch as _decide_batch
        return _decide_batch(self, states, schema, questions=questions,
                             return_details=return_details, min_confidence=min_confidence,
                             **predict_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    system_one = predict

    def route_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        hooks_timeout: Optional[float] = None,
    ) -> List[RouteDecision]:
        """Route a heterogeneous request batch without loading any checkpoints.

        Each request is a mapping with ``state`` and ``questions`` plus the same optional
        routing overrides accepted by :meth:`route`: ``model``, ``task``, ``lang`` and
        ``lang_guess``. The returned decisions preserve input order.

        This is intentionally separate from inference so callers can inspect or aggregate
        routing decisions before paying model-load cost.

        Args:
            requests: Sequence of request dictionaries, each requiring ``state`` and
                ``questions``.
            hooks_timeout: Override the Router's ``hooks_timeout`` for this call, applied to
                every request's ``on_route`` dispatch, as on :meth:`route`.
        """
        if not isinstance(requests, SequenceABC) or isinstance(requests, (str, bytes)):
            raise TypeError("requests must be a sequence of request dictionaries")

        decisions: List[RouteDecision] = []
        for i, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError("request %d must be a dict, got %s" % (i, type(request).__name__))
            if "state" not in request:
                raise ValueError("request %d is missing required key 'state'" % i)
            if "questions" not in request:
                raise ValueError("request %d is missing required key 'questions'" % i)

            questions = request["questions"]
            if not isinstance(questions, dict):
                raise TypeError(
                    "request %d 'questions' must be a dict, got %s"
                    % (i, type(questions).__name__)
                )

            decisions.append(
                self.route(
                    request["state"],
                    questions,
                    model=request.get("model"),
                    task=request.get("task"),
                    lang=request.get("lang"),
                    lang_guess=request.get("lang_guess"),
                    hooks_timeout=hooks_timeout,
                )
            )

        return decisions

    def predict_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        batch_size: Optional[int] = None,
        hooks_timeout: Optional[float] = None,
        min_confidence: Optional[float] = None,
        sort_by_length: bool = False,
    ) -> List[Dict[str, Any]]:
        """Route and execute a heterogeneous request batch with minimal model churn.

        Requests are routed first and grouped by checkpoint. Within each checkpoint,
        requests that share the same question schema are passed to
        ``Agent.predict_batch`` so their states can share forward passes. Results are
        then restored to the original request order.

        Requests may independently specify ``model``, ``task``, ``lang``,
        ``lang_guess``, ``max_len`` or ``head_max_len`` and may use different question schemas.
        ``max_len`` / ``head_max_len`` are the per-request form of the token-budget override
        ``predict`` takes as call arguments: they set the checkpoint's state and question-head
        budgets for that one request, so a wide question can be asked without shrinking the
        batch's other requests to the same window. Requests that ask for different budgets are
        split into separate forward passes, since one ``Agent.predict_batch`` call carries one
        budget for all its states. A start hook may still replace either value on ``ctx``.

        Router-level predict hooks run per request, as ``predict`` runs them: each request
        gets its own ``PredictContext``, so ``on_predict_start`` can replace that request's
        state, questions or token budget, or ``ctx.skip(...)`` it, and ``on_predict_end``
        sees and may replace its result. Requests are grouped for the forward pass after
        their start hooks have run, and a checkpoint group's requests end in reverse of the
        order they started. If a checkpoint group fails, every request of it whose start hook
        ran fails with the exception, a cache hit included: each gets ``on_error`` and then
        ``on_predict_end`` before the exception propagates.

        Args:
            requests: Sequence of request dictionaries. Every item requires ``state`` and
                ``questions`` and may include ``model``, ``task``, ``lang`` or
                ``lang_guess`` routing overrides and ``max_len`` / ``head_max_len`` token-budget
                overrides.
            batch_size: Optional maximum number of states per Agent forward-pass batch.
            hooks_timeout: Override the Router's ``hooks_timeout`` for this call.
            sort_by_length: Forwarded to every ``Agent.predict_batch`` call, so each question
                group pads to a shorter maximum; see ``Agent.predict_batch``. Results retain the
                input order either way. Silently dropped for an attached agent whose
                ``predict_batch`` predates the knob (#294).

        Returns:
            One normal Router prediction result per request, in the same order as the input.
        """
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
        decisions = self.route_batch(requests, hooks_timeout=hooks_timeout)
        if not decisions:
            return []

        # Dict insertion order preserves the order in which model groups first appear. This
        # keeps cache effects deterministic while collapsing an arbitrarily interleaved
        # workload to at most one load per routed checkpoint for this call.
        groups: Dict[str, List[int]] = {}
        for i, decision in enumerate(decisions):
            # Indexed, not `.model`: an on_route hook may replace the decision with a plain dict,
            # which `predict` accepts too. Normalised, because such a hook may name the checkpoint
            # by an alias ("ml"), and that request must share its checkpoint's forward pass.
            groups.setdefault(self.resolve(decision["model"]), []).append(i)

        # Counted rather than marked with None: an end hook may leave a None result, which
        # `predict` returns as it is.
        results: List[Any] = [None] * len(requests)
        answered = 0
        # `compose_hooks`, not `list(self.hooks)`: this is the composition `predict` uses at its
        # own dispatch site, and it is what merges in `set_default_hooks`. Reading the instance
        # list alone silently dropped every process-wide default from the batched path while
        # keeping them on `predict`, so a default audit or metrics hook saw no Router-level event
        # for a request that arrived through `predict_batch`.
        active = compose_hooks(self.hooks)
        raise_errors = self.hooks_raise
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)

        for model_name, indices in groups.items():
            agent = self.load(model_name)
            started: List[PredictContext] = []
            try:
                # One context per request, built and started the way `predict` does it, so a
                # start hook sees -- and can redact, rewrite or skip -- each request before it
                # joins a shared forward pass. A request's own `max_len` / `head_max_len` seed the
                # context here; the grouping below already splits on those fields, so this is the
                # only place the per-request budget can enter.
                for i in indices:
                    ctx = PredictContext(states=[requests[i]["state"]], questions=requests[i]["questions"],
                                         decision=dict(decisions[i]), model=model_name, agent=agent,
                                         router=self,
                                         max_len=requests[i].get("max_len"),
                                         head_max_len=requests[i].get("head_max_len"))
                    started.append(ctx)
                    dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors,
                             lock=self._hooks_lock, timeout=timeout)

                # Agent.predict_batch evaluates one shared question schema and token budget over
                # many states. Preserve Router's heterogeneous-request API by splitting each
                # checkpoint group again on what the start hooks left, so a rewritten question
                # set or `ctx.max_len` only applies to its own request.
                question_groups: List[Dict[str, Any]] = []
                for i, ctx in zip(indices, started):
                    if ctx.results is not None:
                        # A cache hit short-circuits inference; keep the `routing` key predict adds.
                        for result in ctx.results:
                            if isinstance(result, dict):
                                result.setdefault("routing", dict(decisions[i]))
                        continue
                    # Pass token-budget overrides only when set, as `predict` does, so an
                    # Agent-like object that does not accept them still works.
                    overrides = {key: value for key, value in (("max_len", ctx.max_len),
                                                               ("head_max_len", ctx.head_max_len))
                                 if value is not None}
                    # `predict` forwards the language of the request so the agent can apply its
                    # per-language temperatures; the batched path forwarded only the token
                    # budgets, so the same request scored differently depending on the entry
                    # point. Only computed for an agent that actually carries them: `lang` is
                    # otherwise unused, and adding it to the group key would split a group that
                    # shares one forward pass today.
                    lang_key = None
                    if getattr(agent, "lang_temperatures", None):
                        lang_key = requests[i].get("lang")
                        if lang_key is None:
                            detection = decisions[i].get("detection") or {}
                            lang_key = detection.get("language")
                    # Order-sensitive at every nesting level (#166): options are positional, so two
                    # equal schemas with different key orders must not share a group.
                    schema = _question_schema(ctx.questions)
                    for group in question_groups:
                        if (group["schema"] == schema and group["overrides"] == overrides
                                and group["lang"] == lang_key):
                            group["items"].append((i, ctx))
                            break
                    else:
                        question_groups.append({
                            "questions": ctx.questions,
                            "schema": schema,
                            "overrides": overrides,
                            "lang": lang_key,
                            "items": [(i, ctx)],
                        })

                for group in question_groups:
                    items = group["items"]
                    batch_kwargs = dict(group["overrides"])
                    if group["lang"] is not None:
                        batch_kwargs["lang"] = group["lang"]
                    if sort_by_length:
                        # Passed only when on, as the token-budget overrides are: agents attached
                        # via `attach()` may predate the knob (#294).
                        batch_kwargs["sort_by_length"] = True
                    skip = _SKIP_DEFAULTS.set(True)
                    try:
                        batch_results = agent.predict_batch(
                            [ctx.states[0] for _, ctx in items],
                            group["questions"],
                            batch_size=batch_size,
                            **batch_kwargs,
                        )
                    except TypeError as e:
                        # Same tolerance `predict` has for an Agent-like object whose
                        # `predict_batch` predates the `lang` (or `sort_by_length`) argument.
                        retried = False
                        for kwarg in ("lang", "sort_by_length"):
                            if batch_kwargs.get(kwarg) is not None and \
                                    "unexpected keyword argument '%s'" % kwarg in str(e):
                                batch_kwargs.pop(kwarg)
                                retried = True
                        if retried:
                            batch_results = agent.predict_batch(
                                [ctx.states[0] for _, ctx in items],
                                group["questions"],
                                batch_size=batch_size,
                                **batch_kwargs,
                            )
                        else:
                            raise
                    finally:
                        _SKIP_DEFAULTS.reset(skip)

                    if len(batch_results) != len(items):
                        raise RuntimeError(
                            "internal error: Agent.predict_batch returned %d results for %d states"
                            % (len(batch_results), len(items))
                        )

                    for (i, ctx), result in zip(items, batch_results):
                        result["routing"] = dict(decisions[i])
                        ctx.results = [result]

                # Usage is summed here, not while ending, so a malformed usage block (a cached
                # payload's, say) fails the group like any other error instead of escaping from
                # the end loop before a single end hook has run.
                usages = [aggregate_usage(ctx.results) for ctx in started]
                for ctx, usage in zip(started, usages):
                    ctx.usage = usage
            except BaseException as exc:
                # The caller gets this exception and no result, so every started request of the
                # group failed with it, including a cache hit or a request whose question group
                # had already run. Each is still ended, so a hook that opens something in start
                # (a span, an in-flight count) always sees the matching end.
                for ctx in started:
                    if ctx.results:
                        apply_confidence_gate(ctx.results, mc)
                self._end_contexts(active, started, raise_errors, timeout, error=exc)
                raise

            for ctx in started:
                if ctx.results:
                    apply_confidence_gate(ctx.results, mc)
            self._end_contexts(active, started, raise_errors, timeout)
            for i, ctx in zip(indices, started):
                results[i] = ctx.results[0]
            answered += len(indices)
            # Drop this group's references before the next load() can evict its checkpoint:
            # eviction's gc.collect() and empty_cache() only free an Agent nothing still holds,
            # and every context of the group holds it as `ctx.agent`.
            agent = ctx = started = question_groups = group = items = None

        # Every input index is assigned exactly once by construction. Keep this assertion local
        # so a future refactor cannot silently return a partially-filled batch.
        if answered != len(requests):
            raise RuntimeError("internal error: batch execution did not produce every result")

        return results

    predict_many = predict_batch

    def _end_contexts(self, active: List[Any], contexts: List[PredictContext], raise_errors: bool,
                      timeout: Optional[float] = None, error: Optional[BaseException] = None) -> None:
        """Finish each request of a batch the way `predict` finishes one.

        `error` is the exception that failed the batch, if one did: every context then fails
        with it and gets `on_error` before its `on_predict_end`, as in `predict`. Contexts end
        in reverse of the order they started, because all of them started before any ends: a
        hook that sets something in start and resets it in end (a contextvar, a tracing
        context) must unwind the last one first. Every context gets its `on_predict_end` even if
        another's hooks raise; the first such failure is raised afterwards. On a failed context,
        a raising hook is chained onto its error instead, as in `predict`. `elapsed_ms` is set
        on all of them first, so one request's never includes another's error or end hooks.
        """
        now = time.perf_counter()
        for ctx in contexts:
            ctx.elapsed_ms = (now - ctx.started_at) * 1000.0
            if error is not None:
                ctx.error = error
        first_error: Optional[BaseException] = None
        for ctx in reversed(contexts):
            events = ("on_error", "on_predict_end") if ctx.error is not None else ("on_predict_end",)
            for event in events:
                try:
                    dispatch(active, event, ctx, raise_errors=raise_errors,
                             lock=self._hooks_lock, timeout=timeout)
                except BaseException as hook_exc:
                    if ctx.error is not None:
                        ctx.error.__context__ = hook_exc
                    elif first_error is None:
                        first_error = hook_exc
        if first_error is not None:
            raise first_error

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (self.loaded, self.max_loaded, self.default)
