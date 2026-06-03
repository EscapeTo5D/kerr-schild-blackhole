"""回归：默认参数（新旋钮全取原版值）下探针渲染应与基线一致。
effect 用例在各工作线任务中追加。"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(os.path.dirname(__file__), "baseline_probe.npy")


def test_default_render_matches_baseline():
    from tools.probe import render_probe
    img = render_probe()
    assert os.path.exists(BASELINE), "基线缺失：先跑 Task 1 生成 tests/baseline_probe.npy"
    base = np.load(BASELINE)
    assert img.shape == base.shape
    diff = np.abs(img - base)
    assert diff.max() < 2e-3, f"max diff {diff.max():.4g} 超过容差（回归被破坏）"


def _probe_subprocess(sets):
    """子进程跑探针（全新编译），返回渲染 ndarray。sets: list['K=V']。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp.close()
    cmd = [sys.executable, os.path.join("tools", "probe.py"), "--out", tmp.name]
    for s in sets:
        cmd += ["--set", s]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return np.load(tmp.name)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def test_disk_warp_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["DISK_WARP=0.8", "DISK_SHEAR=1.2"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"湍流旋钮未生效，max diff 仅 {diff.max():.4g}"
