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
    img = render_probe(overrides={"STARMAP_PATH": None, "STARMAP_ON": 0})
    assert os.path.exists(BASELINE), "基线缺失：先跑 Task 1"
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


def test_disk_warp_and_shear_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["DISK_WARP=0.8", "DISK_SHEAR=1.2"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"湍流旋钮未生效，max diff 仅 {diff.max():.4g}"


def test_disk_filament_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["DISK_FILAMENT=2.5"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"各向异性未生效，max diff 仅 {diff.max():.4g}"


def test_disk_octave_boost_changes_output():
    base = np.load(BASELINE)
    # 注意：派生量 DISK_OCTAVE_BOOST_EFF 在 config import 时定值，需直接覆盖它
    # （覆盖 DISK_OCTAVE_BOOST 不会重算 _EFF）。_apply 会写到 shading 模块。
    img = _probe_subprocess(["DISK_OCTAVE_BOOST_EFF=1"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"额外 octave 未生效，max diff 仅 {diff.max():.4g}"


def test_photon_ring_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["PHOTONRING_ON=1", "PHOTONRING_STRENGTH=3.0"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"光子环未生效，max diff 仅 {diff.max():.4g}"


def test_inner_layers_change_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["INNER_HALO=2.0", "INNER_VEIL=1.5"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"内缘分层未生效，max diff 仅 {diff.max():.4g}"


def test_starmap_changes_sky():
    import pytest
    # 直接查磁盘素材，避免读到被 render_probe 就地改成 None 的 config.STARMAP_PATH（同进程污染）
    asset = os.path.join(ROOT, "assets", "milkyway_eso0932a.jpg")
    if not os.path.exists(asset):
        pytest.skip("星图素材未找到，跳过")
    base = np.load(BASELINE)
    img = _probe_subprocess([])  # 子进程默认 STARMAP_PATH 指向素材 → 星空与程序化基线不同
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"星图底图未改变星空，max diff 仅 {diff.max():.4g}"


def test_tonemap_aces_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["TONEMAP=1"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"ACES tonemap 未生效，max diff 仅 {diff.max():.4g}"


def test_anamorphic_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["ANAMORPHIC_ON=1", "ANAMORPHIC_STRENGTH=1.5"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"anamorphic 未生效，max diff 仅 {diff.max():.4g}"
