"""The command line, end to end: signal -> masks -> resolve, as a user would run them."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import SRC, run_module_tests  # noqa: E402
from test_resolve import write_run  # noqa: E402

import torch  # noqa: E402

MODEL = textwrap.dedent('''
    from torch import nn

    class DA(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.theta1, self.theta2, self.W = nn.Linear(d, 4), nn.Linear(d, 4), nn.Linear(4, 1)
            self.temperature = 1.0

    class M(nn.Module):
        def __init__(self, d=5):
            super().__init__()
            self.input_dim, self.deep_attention = d, DA(d)
            self.mlp = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
''')


def _cli(tmp, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(PYTHONPATH=SRC, HOME=tmp, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "unveil_tilefilter", *args], cwd=tmp, env=env,
                          capture_output=True, text=True, timeout=300)


def test_signal_then_masks():
    with tempfile.TemporaryDirectory() as tmp:
        code = os.path.join(tmp, "code")
        os.makedirs(code)
        open(os.path.join(code, "clfmodel.py"), "w").write(MODEL)
        sys.path.insert(0, code)
        import clfmodel

        torch.manual_seed(0)
        torch.save({"model": clfmodel.M()}, os.path.join(tmp, "clf.pt"))
        feats = os.path.join(tmp, "feats")
        os.makedirs(feats)
        for i, n in enumerate((40, 300, 2000)):
            torch.save(torch.randn(n, 5), os.path.join(feats, f"S{i}.pt"))
        r = _cli(tmp, "signal", "--features-dir", feats, "--checkpoint", "clf.pt",
                 "--model-code", code, "--out", "sig")
        assert r.returncode == 0, r.stderr
        r = _cli(tmp, "masks", "--signals", "sig", "--out", "m/adaptive.npz",
                 "--check-features-dir", feats)
        assert r.returncode == 0, r.stderr
        side = json.load(open(os.path.join(tmp, "m", "adaptive.npz.json")))
        assert side["stats"]["slides"] == 3 and side["stats"]["slides_filtered"] == 2
        with np.load(os.path.join(tmp, "m", "adaptive.npz")) as z:
            assert int((~z["S2"]).sum()) == 660 and z["S0"].all()
        r = _cli(tmp, "masks", "--signals", "sig", "--out", "m/half.npz", "--drop-frac", "0.5",
                 "--filter", "random_matched", "--seed", "4")
        assert r.returncode == 0, r.stderr
        side = json.load(open(os.path.join(tmp, "m", "half.npz.json")))
        assert side["params"]["drop_frac"] == 0.5 and side["seed"] == 4
        r = _cli(tmp, "masks", "--signals", "sig", "--out", "x.npz", "--drop-frac", "1.5")
        assert r.returncode != 0 and "drop_frac" in r.stderr


def test_resolve_table():
    with tempfile.TemporaryDirectory() as tmp:
        write_run(os.path.join(tmp, "base"), auroc=0.80, pvals={"EOddAbs": 0.001})
        write_run(os.path.join(tmp, "ok"), auroc=0.79)
        write_run(os.path.join(tmp, "unfair"), auroc=0.81, pvals={"EOddAbs": 0.01})
        write_run(os.path.join(tmp, "worse"), auroc=0.70)
        with open(os.path.join(tmp, "t.csv"), "w") as fh:
            fh.write("name,run_dir,baseline_dir,baseline_auroc\n")
            for n in ("ok", "unfair", "worse"):
                fh.write(f"{n},{os.path.join(tmp, n)},{os.path.join(tmp, 'base')},\n")
            fh.write(f"ok_by_value,{os.path.join(tmp, 'ok')},,0.80\n")
        r = _cli(tmp, "resolve", "--table", "t.csv", "--out", "res.csv")
        assert r.returncode == 0, r.stderr
        rows = {row["name"]: row for row in csv.DictReader(open(os.path.join(tmp, "res.csv")))}
        assert rows["ok"]["resolved"] == "True" and rows["ok_by_value"]["resolved"] == "True"
        assert rows["unfair"]["resolved"] == "False" and "EOddAbs" in rows["unfair"]["why"]
        assert rows["worse"]["resolved"] == "False" and rows["worse"]["fair"] == "True"


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
