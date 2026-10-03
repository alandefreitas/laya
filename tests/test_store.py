"""Checkpoint store tests: directories become registered checkpoints, manifests are read strictly.

No weights are loaded: `discover` reads names, paths and manifests, and `Router(**router_options(...))`
registers them without building anything. Run: python tests/test_store.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.router import DEFAULT_MODELS, Router  # noqa: E402
from laya.store import MANIFEST, Checkpoint, discover, router_options, scan, store_dir  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def raises(name, fn, exc, needle=None):
    try:
        fn()
    except exc as error:
        check(name, needle is None or needle in str(error), True)
    except Exception as error:  # noqa: BLE001 -- the wrong exception is the finding
        FAIL.append("%s: raised %r, want %s" % (name, error, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))


def skips(name, root, needle):
    """The store skips (not raises on) a directory, and says why."""
    try:
        _, errors = scan(root)
    except Exception as error:  # noqa: BLE001 -- raising is the finding
        FAIL.append("%s: scan raised %r instead of reporting" % (name, error))
        return
    check(name, bool(errors) and needle in " ".join(errors.values()), True)


def _checkpoint(root, name, manifest=None, weights=True):
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    if weights:
        open(os.path.join(path, "model.safetensors"), "wb").close()
        json.dump({"max_len": 1024}, open(os.path.join(path, "rl_agent_config.json"), "w"))
    if manifest is not None:
        with open(os.path.join(path, MANIFEST), "w", encoding="utf-8") as fh:
            fh.write(manifest if isinstance(manifest, str) else json.dumps(manifest))
    return path


# --------------------------------------------------------------------- where the store is
check("store_dir/LAYA_CHECKPOINTS_DIR wins", store_dir({"LAYA_CHECKPOINTS_DIR": "/srv/ckpt", "XDG_DATA_HOME": "/x"}),
      "/srv/ckpt")
check("store_dir/LAYA_CHECKPOINTS_DIR expands ~", store_dir({"LAYA_CHECKPOINTS_DIR": "~/ckpt"}),
      os.path.join(os.path.expanduser("~"), "ckpt"))
check("store_dir/blank LAYA_CHECKPOINTS_DIR is unset", store_dir({"LAYA_CHECKPOINTS_DIR": "  ", "XDG_DATA_HOME": "/x"}),
      "/x/laya/checkpoints")
import laya.store as _store_module  # noqa: E402
check("module docstring documents the faceted manifest form",
      all(w in _store_module.__doc__ for w in ('"options"', '"for"', '"ids"', "faceted")), True)
check("store_dir/XDG_DATA_HOME", store_dir({"XDG_DATA_HOME": "/data"}), "/data/laya/checkpoints")
check("store_dir/default under ~/.local/share", store_dir({}),
      os.path.join(os.path.expanduser("~"), ".local", "share", "laya", "checkpoints"))

# --------------------------------------------------------------------- discovery
with tempfile.TemporaryDirectory() as tmp:
    check("discover/a missing store is empty", discover(os.path.join(tmp, "nope")), {})
    check("discover/an empty store is empty", discover(tmp), {})

    research = _checkpoint(tmp, "research", {
        "description": "Files claims into a brief",
        "signatures": [["section"], ["section", "subsection"]],
        "base": "english", "training": {"rows": 22436},
    })
    bare = _checkpoint(tmp, "bare")                                   # weights, no manifest: not registered
    pushed = _checkpoint(tmp, "pushed", {"source": "acme/laya-tone"}, weights=False)
    pair = _checkpoint(tmp, "pair", {"source": ["acme/laya", "tone"], "name": "tone-v2"}, weights=False)
    _checkpoint(tmp, "notes", weights=False)                           # neither weights nor source
    open(os.path.join(tmp, "README.md"), "w").close()                  # a file, not a directory
    os.makedirs(os.path.join(tmp, ".hidden"))
    open(os.path.join(tmp, ".hidden", "model.safetensors"), "wb").close()

    found = discover(tmp)
    check("discover/names", sorted(found), ["pushed", "research", "tone-v2"])
    check("scan/a directory with weights and no manifest is reported, not registered",
          (sorted(scan(tmp)[1]), "no manifest" in scan(tmp)[1]["bare"]), (["bare"], True))
    check("scan/the report names the directory", bare in scan(tmp)[1]["bare"], True)
    check("scan/a directory with neither weights nor a manifest is ignored", "notes" in scan(tmp)[1], False)
    check("discover/entries are Checkpoints", all(isinstance(c, Checkpoint) for c in found.values()), True)
    check("discover/a manifest without 'source' serves the directory", found["research"].source, research)
    check("discover/the directory is the source by default", found["research"].source, research)
    check("discover/signatures as lists", found["research"].signatures, [["section"], ["section", "subsection"]])
    check("discover/description", found["research"].description, "Files claims into a brief")
    check("discover/the rest of the manifest is kept", found["research"].manifest["training"], {"rows": 22436})
    check("discover/path is the directory", found["research"].path, research)
    check("discover/a source overrides the directory", found["pushed"].source, "acme/laya-tone")
    check("discover/a [repo, subfolder] source is a pair", found["tone-v2"].source, ("acme/laya", "tone"))
    check("discover/the manifest may rename the directory", found["tone-v2"].path, pair)

    options = router_options(found)
    check("router_options/models", sorted(options["models"]), ["pushed", "research", "tone-v2"])
    check("router_options/signatures only where declared", options["signatures"],
          {"research": [["section"], ["section", "subsection"]]})
    check("router_options/empty store registers nothing", router_options({}), {})

    r = Router(**options)
    check("router/registers every entry", sorted(set(r.models) - set(DEFAULT_MODELS)),
          ["pushed", "research", "tone-v2"])
    q = {"section": {"type": "choice", "instructions": "Which section?", "criteria": {"a": "A", "b": "B"}}}
    check("router/signature from the manifest routes", r.route("A claim about transparency.", q).model, "research")
    check("router/model= names a store entry", r.route("x", q, model="tone-v2").model, "tone-v2")
    check("router/registered reports the store", r.registered["research"],
          {"source": research, "signatures": [["section"], ["section", "subsection"]], "description": None})

    # Degradation: a manifest that cannot be read, or claims a taken name, skips that directory only.
    _checkpoint(tmp, "broken", "{not json")
    skips("discover/unreadable manifest raises", tmp, "cannot read")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", "[1, 2]")
    skips("discover/non-object manifest raises", tmp, "JSON object")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"signatures": [[]]})
    skips("discover/empty signature raises", tmp, "signatures")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"signatures": ["section"]})
    skips("discover/a signature that is not a list raises", tmp, "signatures")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"source": 7})
    skips("discover/a non-string source raises", tmp, "source")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"source": ["acme/laya"]})
    skips("discover/a one-element source pair raises", tmp, "source")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"name": ""})
    skips("discover/a blank name raises", tmp, "name")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"description": 3})
    skips("discover/a non-string description raises", tmp, "description")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"name": "research"})
    skips("discover/a name two directories claim raises", tmp, "claimed by both")
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "broken", {"signatures": [["section"]]})
    skips("scan/one id set cannot select two store entries", tmp, "already registered")
    check("scan/the directory sorted first keeps the signature", ("research" in scan(tmp)[0], "broken" in scan(tmp)[0]),
          (False, True))
    check("scan/the loser is reported under its directory", sorted(scan(tmp)[1]), ["bare", "research"])
    Router(**router_options(discover(tmp)))     # what remains registers without a refusal
    os.remove(os.path.join(tmp, "broken", MANIFEST))
    _checkpoint(tmp, "Bad Name", {})
    skips("scan/a store name the router would reject is skipped", tmp, "invalid checkpoint name")
    Router(**router_options(discover(tmp)))

# A directory named after a built-in or an alias cannot re-point it, and a name is stored canonical.
def _only(tmp, name, manifest=None):
    return _checkpoint(tmp, name, {} if manifest is None else manifest)


with tempfile.TemporaryDirectory() as tmp:
    _only(tmp, "ml")
    skips("discover/an alias name is refused", tmp, "built-in")
with tempfile.TemporaryDirectory() as tmp:
    _only(tmp, "english")
    skips("discover/a built-in name is refused", tmp, "built-in")
with tempfile.TemporaryDirectory() as tmp:
    _only(tmp, "auto")
    skips("discover/the routing word auto is refused", tmp, "auto")
with tempfile.TemporaryDirectory() as tmp:
    _only(tmp, "Research")
    check("discover/names are stored canonical", sorted(discover(tmp)), ["research"])
    check("discover/the checkpoint carries the canonical name", discover(tmp)["research"].name, "research")
with tempfile.TemporaryDirectory() as tmp:
    _checkpoint(tmp, "a", {"name": "Papers"})
    _checkpoint(tmp, "b", {"name": "papers"})
    skips("discover/names differing in case collide", tmp, "claimed by both")

# --------------------------------------------------------------------- pinned signatures in manifests
with tempfile.TemporaryDirectory() as tmp:
    ask = "Which section of the brief does this claim belong to?"
    _checkpoint(tmp, "research", {"signatures": [["screen"], {"section": ask, "subsection": None},
                                                 {"depth": "sha1:" + "0" * 40}]})
    found = discover(tmp)
    check("manifest/mapping signatures are kept as dicts", found["research"].signatures,
          [["screen"], {"section": ask, "subsection": None}, {"depth": "sha1:" + "0" * 40}])
    r = Router(**router_options(found))
    q = {"section": {"type": "choice", "instructions": ask, "criteria": {"a": "A"}},
         "subsection": {"type": "choice", "instructions": "anything", "criteria": {"a": "A"}}}
    check("manifest/a pinned signature from a manifest routes", r.route("claim", q).model, "research")
    q["section"]["instructions"] = "A different question?"
    check("manifest/the pin is enforced", r.route("claim", q).model, "english")
    for bad, why in (([{"ids": [1, 2], "instructions": {}}], "described non-text ids"),
                     ([{"ids": ["m"], "instructions": {"m": ""}}], "described blank pin"),
                     ([{"ids": ["m"], "instructions": {"x": ask}}], "described foreign pin")):
        os.remove(os.path.join(tmp, "research", MANIFEST))
        _checkpoint(tmp, "research", {"signatures": bad})
        skips("manifest/refuses %s" % why, tmp, "signatures")
    os.remove(os.path.join(tmp, "research", MANIFEST))
    _checkpoint(tmp, "research", {"signatures": [{"ids": ["m"], "instructions": {"m": ask}}]})
    check("manifest/the described shape is accepted", Router(**router_options(discover(tmp))).route(
        "x", {"m": {"type": "choice", "instructions": ask, "criteria": {"a": "A"}}}).model, "research")
    for bad, why in (([{"section": 7}], "a non-text pin"), ([{}], "an empty mapping"), ([{"": ask}], "a blank id"),
                     (["section"], "a bare string"), ([{"section": ""}], "a blank pin"),
                     ([{"section": "sha1:3f2a"}], "a malformed digest")):
        os.remove(os.path.join(tmp, "research", MANIFEST))
        _checkpoint(tmp, "research", {"signatures": bad})
        skips("manifest/refuses %s" % why, tmp, "signatures")

# --------------------------------------------------------------------- option pins in manifests
with tempfile.TemporaryDirectory() as tmp:
    screen = "Read or skip this reference?"
    _checkpoint(tmp, "research", {"signatures": [
        {"ids": ["research.screen"], "instructions": {"research.screen": screen},
         "options": {"research.screen": ["read", "skip"]}, "for": "read or skip a reference"},
        {"ids": ["research.section"], "instructions": {"research.section": "Which section?"}}]})
    found = discover(tmp)
    r = Router(**router_options(found))
    q = {"research.screen": {"type": "choice", "instructions": screen, "criteria": {"skip": "no", "read": "yes"}}}
    check("manifest/an options pin from a manifest routes", r.route("ref", q).model, "research")
    q["research.screen"]["criteria"]["maybe"] = "later"
    check("manifest/a changed label set falls back to the base model", r.route("ref", q).model, "english")
    check("manifest/the purpose is reported", r.registered["research"]["signatures"][0]["for"], "read or skip a reference")
    for bad, why in (([{"ids": ["m"], "options": {"m": "read"}}], "a bare-string options pin"),
                     ([{"ids": ["m"], "options": {"m": []}}], "an empty options pin"),
                     ([{"ids": ["m"], "options": {"x": ["a"]}}], "an options pin outside the ids"),
                     ([{"ids": ["m"], "for": 3}], "a non-text purpose"),
                     ([{"ids": ["m"], "options": {"m": ["a"]}, "purpose": "x"}], "the Signature keyword 'purpose'"),
                     ([{"ids": "m", "options": {"m": ["a"]}}], "ids given as a string"),
                     ([{"ids": ["m"], "options": None}], "a null options facet"),
                     ([{"ids": [], "options": {}}], "empty ids")):
        os.remove(os.path.join(tmp, "research", MANIFEST))
        _checkpoint(tmp, "research", {"signatures": bad})
        skips("manifest/refuses %s" % why, tmp, "signatures")

# --------------------------------------------------------------------- the manifest marks completion
with tempfile.TemporaryDirectory() as tmp:
    _checkpoint(tmp, "research", None)                       # weights landed, manifest not yet written
    check("scan/an export still being written is not registered", (scan(tmp)[0], list(scan(tmp)[1])), ({}, ["research"]))
    with open(os.path.join(tmp, "research", MANIFEST), "w") as fh:
        json.dump({}, fh)
    check("scan/the manifest completes it", (sorted(scan(tmp)[0]), scan(tmp)[1]), (["research"], {}))
    _checkpoint(tmp, "empty", {}, weights=False)
    skips("scan/a manifest with neither weights nor a source is reported", tmp, "no weights")
    check("scan/...and not registered", sorted(scan(tmp)[0]), ["research"])

# A store directory that cannot be listed is the one thing that still raises.
with tempfile.TemporaryDirectory() as tmp:
    store = os.path.join(tmp, "store")
    _checkpoint(store, "research", {})
    os.chmod(store, 0)
    try:
        if os.access(store, os.R_OK):       # running as root: permissions do not apply
            check("scan/an unlistable store raises (skipped as root)", True, True)
        else:
            raises("scan/an unlistable store raises", lambda: scan(store), ValueError, "cannot list")
    finally:
        os.chmod(store, 0o755)

# `keep`: a conflict refuses the new or edited manifest, never the unchanged checkpoint being served,
# whatever the directory order (the reload case).
def _keep_case():
    root = tempfile.mkdtemp()
    for name, sigs in (("alpha", [["s2"]]), ("beta", [["s3"]])):
        os.makedirs(os.path.join(root, name))
        with open(os.path.join(root, name, "checkpoint.json"), "w") as fh:
            json.dump({"source": "org/" + name, "signatures": sigs}, fh)
    served, _ = scan(root)
    with open(os.path.join(root, "alpha", "checkpoint.json"), "w") as fh:
        json.dump({"source": "org/alpha", "signatures": [["s3"]]}, fh)      # edited onto beta's signature
    plain, plain_errors = scan(root)
    kept, kept_errors = scan(root, keep=served)
    return (sorted(plain), sorted(plain_errors), sorted(kept), sorted(kept_errors))


check("scan/keep protects the unchanged checkpoint from an edited one", _keep_case(),
      (["alpha"], ["beta"], ["beta"], ["alpha"]))

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all store tests passed")
sys.exit(1 if FAIL else 0)

