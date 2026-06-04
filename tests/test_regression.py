"""回归：把所有增强旋钮 override 回「原版/关闭」值后，探针渲染应与基线一致。

注意：config.py 的活跃值现在是电影化 look（DISK_WARP/INNER_GLOW/WHITE_* 等非原版），
所以基线测试不能直接拿 config 活跃值比，而是显式把全部增强旋钮压回原版（_ORIG_BASELINE），
测的是「代码在原版参数下能逐像素复现基线」——即增强是叠加式、逐项可关。
effect 用例在原版基线之上只开一个旋钮，验证该旋钮确实改变输出。"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(os.path.dirname(__file__), "baseline_probe.npy")

# 全部增强旋钮的「原版/关闭」值。任一取此值即复刻 lstSRS 原观感；全取即基线。
_ORIG_BASELINE = {
    "FOV_FACTOR": 1.2,          # 基线抓取时的相机视野（config 活跃值已改 1.4）；探针在调用处读 config.FOV_FACTOR
    "DISK_WARP": 0.0,
    "DISK_SHEAR": 0.0,
    "DISK_FILAMENT": 1.0,
    "DISK_OCTAVE_BOOST_EFF": 0,
    "INNER_HALO": 0.0,
    "INNER_VEIL": 0.0,
    "INNER_GLOW": 1.0,
    "WHITE_BASE_POW": 4.0,
    "WHITE_SIZE": 0.0,
    "PHOTONRING_ON": 0,
    "STARMAP_PATH": None,
    "STARMAP_ON": 0,
}


def test_default_render_matches_baseline():
    from tools.probe import render_probe
    img = render_probe(overrides=dict(_ORIG_BASELINE))
    assert os.path.exists(BASELINE), "基线缺失：先跑 Task 1"
    base = np.load(BASELINE)
    assert img.shape == base.shape
    diff = np.abs(img - base)
    assert diff.max() < 2e-3, f"max diff {diff.max():.4g} 超过容差（回归被破坏）"


def _baseline_sets():
    """原版基线的 --set 列表（供子进程先铺平到原版，再叠单个 effect 旋钮）。"""
    return [f"{k}={v}" for k, v in _ORIG_BASELINE.items()]


def _probe_subprocess(sets):
    """子进程跑探针（全新编译）：先压回原版基线，再叠本用例旋钮，返回渲染 ndarray。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp.close()
    cmd = [sys.executable, os.path.join("tools", "probe.py"), "--out", tmp.name]
    for s in _baseline_sets() + sets:          # 基线在前、effect 旋钮在后覆盖
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
    # 派生量 DISK_OCTAVE_BOOST_EFF 在 config import 时定值，直接覆盖它（_apply 写到 shading 模块）
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


def test_white_size_changes_output():
    base = np.load(BASELINE)
    # 解耦亮度函数：基础幂调小 + 截断墙 + 内缘提亮，白热盘体应明显改变输出
    img = _probe_subprocess(["WHITE_BASE_POW=2.0", "WHITE_SIZE=0.001", "INNER_GLOW=3.0"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"白热亮度函数未生效，max diff 仅 {diff.max():.4g}"


def test_starmap_changes_sky():
    import pytest
    asset = os.path.join(ROOT, "assets", "milkyway_eso0932a.jpg")
    if not os.path.exists(asset):
        pytest.skip("星图素材未找到，跳过")
    base = np.load(BASELINE)
    # 原版基线（程序化天空）之上单独开星图底图 → 天空应不同
    img = _probe_subprocess([f"STARMAP_PATH={asset}", "STARMAP_ON=1"])
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


def test_vignette_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["VIGNETTE_STRENGTH=0.8"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"暗角未生效，max diff 仅 {diff.max():.4g}"


def test_grain_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["GRAIN_STRENGTH=0.15"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"颗粒未生效，max diff 仅 {diff.max():.4g}"


def test_chroma_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["CHROMA_AB_STRENGTH=3.0"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"色散未生效，max diff 仅 {diff.max():.4g}"
