"""Step 1: the per-tile demographic signal.

For every slide, an attention-MIL demographic classifier (trained by you, to predict the
binary sensitive attribute from the slide) gives three things the filter reads:

    attention   (n,) its attention over the slide's tiles (softmax weights)
    tile_score  (n,) the probability of group 1 it gives each tile on its own, in [0, 1]
    slide_pred  ()   its slide-level probability of group 1, in [0, 1]

They are stored one file per slide, ``<signal_dir>/<folder_id>.npz``, where ``folder_id``
is the stem of the slide's feature file. The tile order must be the order of the rows of
that feature file, because the masks index those rows.

Two ways to compute them:

* ``gated_abmil`` (default): the gated-attention ABMIL classifier used in the paper. The
  model object must have ``deep_attention.theta1``, ``deep_attention.theta2``,
  ``deep_attention.W``, ``deep_attention.temperature``, ``mlp`` (ending in a sigmoid, so
  its outputs are probabilities) and ``input_dim``. The computation below is the one that
  produced the paper's signals, chunked so a large slide never holds more than ``chunk`` x
  ``input_dim`` intermediate values.
* any other model: pass ``--adapter module:function``. ``function(model, x)`` gets the
  slide's (n, d) float32 tensor and returns ``(attention, tile_score, slide_pred)``.

You can also compute the signal elsewhere and write the same .npz files yourself
(``save_signal``): the masks step reads nothing else, and checks each file on loading as
``save_signal`` does (probabilities in [0, 1], non-negative attention, finite values).

A signal directory holds one classifier's output only. ``signal_manifest.json`` names the
classifier (checkpoint md5) and the settings, and is written before the first slide.
"""

import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np

from . import __version__
from .filters import SignalBundle

SIGNAL_KEYS = ("attention", "tile_score", "slide_pred", "n_tiles")
MANIFEST = "signal_manifest.json"
FEATURE_EXTS = (".pt", ".npy", ".h5")
# The fields that must agree before a run may add slides to an existing signal directory:
# mixing two classifiers (or two settings) in one directory would make the masks
# inconsistent from slide to slide.
MANIFEST_IDENTITY = ("checkpoint_md5", "checkpoint_key", "adapter", "chunk", "dtype",
                     "activation", "feature_ext", "h5_key")


class SignalError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# reading and writing
# --------------------------------------------------------------------------------------

def signal_path(signal_dir, folder_id):
    return os.path.join(signal_dir, f"{folder_id}.npz")


def list_signal_ids(signal_dir):
    """Sorted folder_ids that have a signal file in ``signal_dir``."""
    if not os.path.isdir(signal_dir):
        raise SignalError(f"no signal directory {signal_dir}")
    return sorted(f[:-4] for f in os.listdir(signal_dir) if f.endswith(".npz"))


def save_signal(path, attention, tile_score, slide_pred, dtype="float16"):
    """Write one slide's signal. float16 is what the paper stored; float32 keeps more."""
    if dtype not in ("float16", "float32"):
        raise ValueError(f"dtype must be float16 or float32, got {dtype!r}")
    att = np.asarray(attention)
    tile = np.asarray(tile_score)
    validate_signal(att, tile, slide_pred, os.path.basename(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, attention=att.astype(dtype), tile_score=tile.astype(dtype),
                            slide_pred=np.float32(slide_pred),
                            n_tiles=np.int32(att.shape[0]))
    os.replace(tmp, path)
    return path


def load_signal(signal_dir, folder_id):
    """One slide's SignalBundle (arrays as float32), checked as save_signal checks it, so a
    file written by other code with, e.g., logits instead of probabilities is refused."""
    path = signal_path(signal_dir, folder_id)
    if not os.path.exists(path):
        raise SignalError(f"no demographic signal for slide {folder_id!r} at {path}")
    with np.load(path) as d:
        missing = [k for k in SIGNAL_KEYS if k not in d.files]
        if missing:
            raise SignalError(f"{path} lacks {missing}")
        validate_signal(d["attention"], d["tile_score"], float(d["slide_pred"]), path)
        return SignalBundle(d["attention"], d["tile_score"], float(d["slide_pred"]),
                            int(d["n_tiles"]), folder_id)


def validate_signal(attention, tile_score, slide_pred, name=""):
    att = np.asarray(attention, dtype=np.float64)
    tile = np.asarray(tile_score, dtype=np.float64)
    if att.ndim != 1 or att.shape != tile.shape:
        raise SignalError(f"{name}: attention {att.shape} and tile_score {tile.shape} must be "
                          f"1-D and of equal length")
    if not (np.isfinite(att).all() and np.isfinite(tile).all()
            and np.isfinite(float(slide_pred))):
        raise SignalError(f"{name}: the signal has non-finite values")
    if (att < 0).any():
        raise SignalError(f"{name}: attention has negative values; expected softmax weights")
    if (tile < 0).any() or (tile > 1).any() or not 0.0 <= float(slide_pred) <= 1.0:
        raise SignalError(f"{name}: tile_score and slide_pred must be probabilities in "
                          f"[0, 1]; if your model returns logits, use --activation sigmoid")


# --------------------------------------------------------------------------------------
# features and classifier
# --------------------------------------------------------------------------------------

def load_features(path, h5_key="features"):
    """One slide's (n, d) float32 torch tensor from .pt, .npy or .h5.

    .pt is read the way the paper's pipeline reads it (a (1, n, d) tensor is squeezed);
    .h5 reads the dataset ``h5_key`` (CLAM-style files keep it under "features")."""
    import torch

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        x = torch.load(path, map_location="cpu", weights_only=True)
    elif ext == ".npy":
        x = torch.from_numpy(np.load(path))
    elif ext == ".h5":
        import h5py

        with h5py.File(path, "r") as fh:
            if h5_key not in fh:
                raise SignalError(f"{path} has no dataset {h5_key!r}; pass --h5-key")
            x = torch.from_numpy(fh[h5_key][()])
    else:
        raise SignalError(f"{path}: unsupported feature file type {ext!r}; use {FEATURE_EXTS}")
    if x.ndim == 3:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise SignalError(f"{path}: expected an (n, d) feature matrix, got shape "
                          f"{tuple(x.shape)}")
    if x.dtype != torch.float32:
        x = x.float()
    return x


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_classifier(checkpoint, code_dirs=(), key="model", model_config=None):
    """Load a demographic classifier.

    ``code_dirs`` go on sys.path first, so the model's class can be found. With
    ``model_config``, the checkpoint is a plain state_dict (the released files): the JSON
    names the class, ``model_type``, in the ``model`` module of ``code_dirs`` (or as
    ``module:Class``), and its ``model_init_args``; the weights are loaded with
    ``weights_only=True``. Without it, the checkpoint is unpickled as a whole object, and
    ``key`` picks the model out of a saved dict (the paper's own checkpoints keep it under
    "model"; pass "" for a bare object): load only files you trust."""
    import torch

    for d in reversed(list(code_dirs)):
        if d not in sys.path:
            sys.path.insert(0, d)
    if model_config:
        with open(model_config) as fh:
            cfg = json.load(fh)
        module_name, _, class_name = cfg["model_type"].rpartition(":")
        cls = getattr(importlib.import_module(module_name or "model"), class_name)
        obj = cls(**cfg.get("model_init_args", {}))
        obj.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        obj.eval()
        return obj
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if key:
        if not isinstance(obj, dict) or key not in obj:
            raise SignalError(f"{checkpoint}: no entry {key!r}; pass --checkpoint-key")
        obj = obj[key]
    if not hasattr(obj, "eval"):
        raise SignalError(f"{checkpoint}: the loaded object is a {type(obj).__name__}, not a "
                          f"model. A state_dict alone cannot be used: load the model in "
                          f"your own --adapter.")
    obj.eval()
    return obj


def _gated_abmil_compat(model):
    """The paper's checkpoints predate four attributes their forward() reads."""
    for attr, default in (("input_batch_norm_features", []), ("batch_norm", None),
                          ("input_dropout_rate", 0.0), ("attention_dropout_rate", 0.0)):
        if not hasattr(model, attr):
            setattr(model, attr, default)
    return model


def gated_abmil_signal(model, x, chunk=2048):
    """(attention, tile_score, slide_pred) of the gated-attention ABMIL classifier.

        A   = softmax(W(tanh(theta1 x) * sigmoid(theta2 x)) / temperature)
        bag = sum_i A_i x_i ;  slide_pred = mlp(bag) ;  tile_score_i = mlp(x_i)

    The arithmetic and the chunking are those that produced the paper's signals (chunk
    2048, one thread); a different chunk size changes the float sums in the last bits.
    """
    import torch

    with torch.inference_mode():
        n = x.shape[0]
        da = model.deep_attention
        logits = torch.empty(n, dtype=torch.float32)
        for i in range(0, n, chunk):
            xc = x[i:i + chunk]
            logits[i:i + chunk] = da.W(torch.tanh(da.theta1(xc))
                                       * torch.sigmoid(da.theta2(xc))).squeeze(-1)
        att = torch.softmax(logits / da.temperature, dim=0)
        del logits
        bag = torch.zeros(model.input_dim, dtype=torch.float32)
        tile = torch.empty(n, dtype=torch.float32)
        for i in range(0, n, chunk):
            xc = x[i:i + chunk]
            bag += (xc * att[i:i + chunk, None]).sum(0)
            tile[i:i + chunk] = model.mlp(xc).squeeze(-1)
        pred = float(model.mlp(bag.unsqueeze(0)).ravel()[0])
    return att.numpy(), tile.numpy(), pred


def load_adapter(spec):
    """``gated_abmil`` or ``module:function`` (importable once --model-code is on sys.path)."""
    if spec == "gated_abmil":
        return None
    if ":" not in spec:
        raise SignalError(f"--adapter {spec!r}: use gated_abmil or module:function")
    mod, fn = spec.split(":", 1)
    f = getattr(importlib.import_module(mod), fn, None)
    if not callable(f):
        raise SignalError(f"--adapter {spec!r}: {fn} is not a callable in {mod}")
    return f


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=np.float64)))


def compute_one(model, x, adapter=None, chunk=2048, activation="none"):
    if adapter is None:
        att, tile, pred = gated_abmil_signal(model, x, chunk=chunk)
    else:
        att, tile, pred = adapter(model, x)
        att = np.asarray(att.detach().cpu() if hasattr(att, "detach") else att)
        tile = np.asarray(tile.detach().cpu() if hasattr(tile, "detach") else tile)
        pred = float(pred)
    if activation == "sigmoid":
        tile, pred = _sigmoid(tile).astype(np.float32), float(_sigmoid(pred))
    elif activation != "none":
        raise ValueError(f"activation must be none or sigmoid, got {activation!r}")
    if np.asarray(att).shape[0] != x.shape[0]:
        raise SignalError(f"the adapter returned {np.asarray(att).shape[0]} attention values "
                          f"for {x.shape[0]} tiles")
    return att, tile, pred


# --------------------------------------------------------------------------------------
# the signal step
# --------------------------------------------------------------------------------------

def feature_files(features_dir, feature_ext=".pt", slides=None):
    """[(folder_id, path)] of the slides to process, sorted by folder_id."""
    if feature_ext not in FEATURE_EXTS:
        raise SignalError(f"--feature-ext must be one of {FEATURE_EXTS}")
    if not os.path.isdir(features_dir):
        raise SignalError(f"no feature directory {features_dir}")
    found = {f[:-len(feature_ext)]: os.path.join(features_dir, f)
             for f in os.listdir(features_dir) if f.endswith(feature_ext)}
    if slides is not None:
        want = list(dict.fromkeys(slides))
        missing = [s for s in want if s not in found]
        if missing:
            raise SignalError(f"{len(missing)} listed slide(s) have no {feature_ext} file in "
                              f"{features_dir}, e.g. {missing[:3]}")
        found = {s: found[s] for s in want}
    return sorted(found.items())


def _manifest_identity(m):
    return {k: m.get(k) for k in MANIFEST_IDENTITY}


def _write_manifest(mpath, manifest):
    tmp = f"{mpath}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, mpath)


def compute_signals(features_dir, checkpoint, out_dir, code_dirs=(), checkpoint_key="model",
                    adapter="gated_abmil", feature_ext=".pt", h5_key="features", slides=None,
                    chunk=2048, threads=1, dtype="float16", activation="none",
                    overwrite=False, log=print, model_config=None):
    """Compute and store the signal of every slide; returns the manifest dict.

    The manifest (classifier md5 and settings) is written BEFORE the first slide, so a run
    that is killed still names what wrote its files. Slides whose file exists are kept
    unless ``overwrite``. A directory written with another classifier or other settings --
    or holding signal files but no manifest -- is refused; with ``overwrite`` it is
    recomputed whole, and only if this run covers every slide in it, so one directory
    never mixes two classifiers."""
    import torch

    torch.set_num_threads(int(threads))
    model = load_classifier(checkpoint, code_dirs, checkpoint_key, model_config)
    fn = load_adapter(adapter)
    if fn is None:
        _gated_abmil_compat(model)
        for attr in ("deep_attention", "mlp", "input_dim"):
            if not hasattr(model, attr):
                raise SignalError(f"the classifier has no {attr!r}: it is not the gated "
                                  f"ABMIL the default adapter expects; pass --adapter")
    manifest = {
        "package": "unveil_tilefilter", "version": __version__,
        "checkpoint": os.path.abspath(checkpoint), "checkpoint_md5": md5_file(checkpoint),
        "checkpoint_key": checkpoint_key if not model_config else None,
        "model_config": os.path.abspath(model_config) if model_config else None,
        "adapter": adapter, "chunk": int(chunk),
        "threads": int(threads), "dtype": dtype, "activation": activation,
        "features_dir": os.path.abspath(features_dir), "feature_ext": feature_ext,
        "h5_key": h5_key if feature_ext == ".h5" else None, "code_dirs": list(code_dirs),
        "torch": torch.__version__, "numpy": np.__version__,
    }
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, MANIFEST)
    todo = feature_files(features_dir, feature_ext, slides)
    existing = set(list_signal_ids(out_dir))
    old = None
    if os.path.exists(mpath):
        with open(mpath) as fh:
            old = json.load(fh)
    if existing and (old is None or _manifest_identity(old) != _manifest_identity(manifest)):
        why = ("has signal files but no manifest" if old is None else
               f"was written with other settings ({_manifest_identity(old)})")
        if not overwrite:
            raise SignalError(f"{out_dir} {why}; use a new --out, or --overwrite to "
                              f"recompute every slide in it")
        others = sorted(existing - {fid for fid, _ in todo})
        if others:
            raise SignalError(f"{out_dir} {why}, and {len(others)} of its slides are not in "
                              f"this run (e.g. {others[:3]}), so --overwrite would leave two "
                              f"classifiers' signals side by side; use a new --out")
        for fid in sorted(existing):
            os.remove(signal_path(out_dir, fid))
    manifest.update(status="running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    _write_manifest(mpath, manifest)
    done = kept = 0
    failed = []
    t0 = time.time()
    for folder_id, path in todo:
        out = signal_path(out_dir, folder_id)
        if os.path.exists(out) and not overwrite:
            kept += 1
            continue
        try:
            x = load_features(path, h5_key)
            if fn is None and x.shape[-1] != model.input_dim:
                raise SignalError(f"feature dim {x.shape[-1]} != classifier input_dim "
                                  f"{model.input_dim}")
            att, tile, pred = compute_one(model, x, fn, chunk, activation)
            save_signal(out, att, tile, pred, dtype)
            done += 1
        except Exception as e:                                    # noqa: BLE001
            failed.append({"folder_id": folder_id, "error": f"{type(e).__name__}: {e}"[:300]})
            log(f"[signal] FAILED {folder_id}: {failed[-1]['error']}")
    manifest.update(status="finished", slides_computed=done, slides_kept=kept,
                    slides_failed=len(failed), failures=failed,
                    seconds=round(time.time() - t0, 1),
                    written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    _write_manifest(mpath, manifest)
    log(f"[signal] {out_dir}: computed {done}, kept {kept}, failed {len(failed)}")
    return manifest
