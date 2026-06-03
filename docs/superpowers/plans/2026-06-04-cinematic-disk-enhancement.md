# 吸积盘电影化增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改物理与积分器的前提下，给 Kerr-Schild 黑洞渲染器做一轮电影化视觉增强：盘体丝状湍流、光子环+内缘分层、图像底图星空、后期电影化。

**Architecture:** 沿用现有模块边界（config 常量 → fields 缓冲/贴图 → geodesic 积分 → shading 盘/天 → render kernel → postprocess → main）。所有新行为经 `config.py` 旋钮暴露，**默认值等于原版/关闭**，保证逐项可关、可逐像素回归。GUI 实时档经 `ti.static` 派生标志自动降级。

**Tech Stack:** Python 3 + Taichi（GPU，`ti.f32`）；NumPy（探针/回归对比）；pytest（回归 + 子进程 effect 测试）；外部素材为一张等距柱状银河星图（NASA SVS Deep Star Maps 2020，公有领域）。

**验证策略（对 TDD 的领域适配）：** 经典单元断言不适配 GPU 画面，且 config 在 import 时 `ti.init` 并把常量烘进 kernel。因此自动化验证锚定 spec 的两条保证：
- **回归测试**：默认参数（新旋钮全取原版值）下，低分辨率探针渲染与基线 `tests/baseline_probe.npy` 逐像素一致（容差 2e-3）。同进程直接调用 `render_probe()`。
- **effect 测试**：通过子进程跑探针并用 `--set` 覆盖旋钮（每次全新 Taichi 编译），断言输出相对基线的差异超过阈值，证明旋钮确实生效。
- **肉眼复核**：每条工作线全开时出一张探针 PNG 人工确认观感。

---

## 文件结构

**新建：**
- `tools/__init__.py` —— 让 `tools` 可作为包被 pytest import。
- `tools/probe.py` —— 确定性低分辨率探针：固定相机/时间/分辨率渲染一帧，支持 `--set k=v` 覆盖旋钮、`--out` 指定输出。回归与 effect 测试共用。
- `tests/__init__.py`、`tests/conftest.py` —— 把仓库根加入 `sys.path`。
- `tests/test_regression.py` —— 回归 + effect 测试。
- `tests/baseline_probe.npy` —— 原版基线（Task 1 从未改动的代码生成并提交）。
- `assets/starmap_2020_4k.tif`（或 png）—— 等距柱状银河星图（Task C 下载）。
- `assets/STARMAP_SOURCE.md` —— 星图来源与授权记录。

**修改：**
- `config.py` —— 四条工作线的新旋钮 + 末尾派生标志（`STARMAP_ON` / `DISK_OCTAVE_BOOST_EFF`）。
- `shading.py` —— A（域扭曲/剪切/各向异性/octave）、B（内缘 halo/veil 两层）、C（`sky_color` 拆出 `_proc_sky` + 星图采样）。
- `render.py` —— B（积分循环维护 `r_min`，循环后叠加光子环）。
- `fields.py` —— C（`starmap_tex` field + `load_starmap_tex`）。
- `main.py` —— C（加载星图）。
- `postprocess.py` —— D（tonemap 选择 + anamorphic 拉丝 + vignette/grain/chroma）。

**任务顺序：** Task 0（探针）→ Task 1（基线）必须最先。之后 A/B/C/D（Task 2~5）相互独立、可任意次序，各自独立提交。Task 6 收尾出对比图。

---

## Task 0: 探针 + 回归/effect 测试骨架

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/probe.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_regression.py`

- [ ] **Step 1: 建 `tools/__init__.py`（空包标记）**

```python
```
（空文件即可。Windows PowerShell：`New-Item -ItemType File tools/__init__.py`）

- [ ] **Step 2: 写探针 `tools/probe.py`**

```python
"""确定性低分辨率探针：固定相机/时间/分辨率渲染一帧，供回归与 effect 测试复用。

绕过 config 的高分辨率：在 kernel 编译前改写各模块已 import 的 RES 副本
（Taichi kernel/func 惰性编译，首次 render 调用前改这些 module 全局即可生效）。

用法：
    python tools/probe.py --out probe.npy
    python tools/probe.py --out probe_warp.npy --set DISK_WARP=0.8 --set DISK_SHEAR=1.0
"""
import argparse
import numpy as np
import taichi as ti

import config
import fields
import render
import shading
import postprocess

PROBE_RES = (320, 180)
PROBE_PHI = 0.0
PROBE_THETA = 0.08
PROBE_T = 0.2

_MODULES = (config, fields, render, shading, postprocess)


def _coerce(val: str):
    """'0.8'->0.8, '2'->2, '(1.0,0.9,0.7)'->tuple, 其它原样字符串。"""
    s = val.strip()
    if s.startswith("(") and s.endswith(")"):
        return tuple(_coerce(x) for x in s[1:-1].split(",") if x.strip())
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.lower() in ("none", "null"):
        return None
    return s


def _apply(name, value):
    """把覆盖值写到所有持有该名字副本的模块（编译前生效）。"""
    for m in _MODULES:
        if hasattr(m, name):
            setattr(m, name, value)


def render_probe(overrides=None):
    # 先降分辨率（务必在首次 render 编译前）
    for m in (config, fields, render):
        if hasattr(m, "RES_X"):
            m.RES_X, m.RES_Y = PROBE_RES
        if hasattr(m, "SAMPLES"):
            m.SAMPLES = 1
    if overrides:
        for k, v in overrides.items():
            _apply(k, v)
    fields.build()
    fields.load_noise_tex(config.NOISE_TEX_PATH or None)
    fields.load_organic_tex(config.ORGANIC_TEX_PATH or None)
    sm = getattr(config, "STARMAP_PATH", None)
    if sm and hasattr(fields, "load_starmap_tex"):
        fields.load_starmap_tex(sm)
    render.render_frame(PROBE_PHI, PROBE_THETA, PROBE_T, samples=1)
    return fields.disp.to_numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe.npy")
    ap.add_argument("--set", action="append", default=[], dest="sets")
    args = ap.parse_args()
    ov = {}
    for item in args.sets:
        k, v = item.split("=", 1)
        ov[k.strip()] = _coerce(v)
    img = render_probe(ov)
    np.save(args.out, img)
    if args.out.endswith(".npy"):
        ti.tools.imwrite(fields.disp, args.out[:-4] + ".png")
    print(f"probe saved: {args.out} shape={img.shape} overrides={ov}")
```

- [ ] **Step 3: 建 `tests/__init__.py`（空）与 `tests/conftest.py`**

`tests/conftest.py`：

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 4: 写 `tests/test_regression.py`（先只放回归用例，effect 用例后续工作线再加）**

```python
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
    subprocess.run(cmd, cwd=ROOT, check=True)
    img = np.load(tmp.name)
    os.unlink(tmp.name)
    return img
```

- [ ] **Step 5: 运行测试，确认现在因缺基线而失败**

Run: `python -m pytest tests/test_regression.py -v`
Expected: `test_default_render_matches_baseline` FAIL，断言信息为"基线缺失"。

- [ ] **Step 6: 提交**

```bash
git add tools/__init__.py tools/probe.py tests/__init__.py tests/conftest.py tests/test_regression.py
git commit -m "test: 加低分辨率探针 + 回归测试骨架"
```

---

## Task 1: 捕获原版基线

在任何功能改动**之前**，从当前未改动的代码生成基线，锁住原版观感。

**Files:**
- Create: `tests/baseline_probe.npy`

- [ ] **Step 1: 生成基线**

Run: `python tools/probe.py --out tests/baseline_probe.npy`
Expected: 打印 `probe saved: tests/baseline_probe.npy shape=(320, 180, 3) ...`，并生成 `tests/baseline_probe.png`。

- [ ] **Step 2: 肉眼确认基线 PNG 是当前正常画面**

打开 `tests/baseline_probe.png`，确认是正常的黑洞+盘+星空缩略图（非全黑/全白）。

- [ ] **Step 3: 运行回归测试，应通过**

Run: `python -m pytest tests/test_regression.py::test_default_render_matches_baseline -v`
Expected: PASS。

- [ ] **Step 4: 提交基线**

```bash
git add tests/baseline_probe.npy tests/baseline_probe.png
git commit -m "test: 提交原版渲染基线（回归锚点）"
```

---

## Task 2: 工作线 A —— 盘体丝状湍流

域扭曲 + 较差自转剪切 + 方位各向异性 + 离线额外 octave。全部默认取原版值。

**Files:**
- Modify: `config.py`（新增 A 旋钮 + 派生 `DISK_OCTAVE_BOOST_EFF`）
- Modify: `shading.py:102-155`（`gas_disc_sample` 噪声块）
- Modify: `tests/test_regression.py`（追加 A 的 effect 用例）

- [ ] **Step 1: config 加 A 旋钮**

在 `config.py` 的"吸积盘"段（`TIME_SPEED` 行之后）插入：

```python
# ---- A 丝状湍流（默认全取原版值）----
DISK_WARP = 0.0          # 域扭曲强度：低频噪声扰动采样坐标，造流体丝状旋涡。0=原版块状
DISK_SHEAR = 0.0         # 较差自转剪切：噪声方位相位随 r 变化（内快外慢），丝拖成螺旋。0=均匀=原版
DISK_FILAMENT = 1.0      # 方位各向异性拉伸比：沿切向拉长结构顺流线。1=各向同性=原版
DISK_OCTAVE_BOOST = 0    # 离线额外高频 octave 数（0/1/2）。0=原版层数；GUI 档强制 0
```

并在文件末尾"模式派生"段（`if INTERACTIVE:` 块之后）追加：

```python
# A：离线才加额外 octave；GUI 实时档强制 0（编译期常量，被 ti.static 用）
DISK_OCTAVE_BOOST_EFF = 0 if INTERACTIVE else DISK_OCTAVE_BOOST
```

- [ ] **Step 2: shading.py import 新名字**

把 `shading.py` 顶部 `from config import (...)` 列表追加：`DISK_WARP, DISK_SHEAR, DISK_FILAMENT, DISK_OCTAVE_BOOST_EFF`。

- [ ] **Step 3: 改 `gas_disc_sample` 噪声坐标构造块**

定位 `shading.py` 中（约 127-143 行）这段：

```python
        if cov0 >= 0.01:
            ang = ti.atan2(y, x)          # 切口在 -x（黑洞背后），避开正前方那条接缝
            rcx = (dc * 1.5 + 0.55) * 0.95
            rcy = (ang * 1.5) * 0.95
            rcz = (dd * 1.5) * 0.95
            sft = tparam * TIME_SPEED
            pa = vec3(rcx, rcy + sft, rcz)
            pb = vec3(rcx, rcy, rcz)
            n1 = (noise3(pa * 3.0) * 0.5 + 0.5) * (noise3(pb * 6.0) * 0.5 + 0.5) \
                * (noise3(pa * 12.0) * 0.5 + 0.5) * (noise3(pb * 24.0) * 0.5 + 0.5)
            pc = vec3(rcx + 30.0, rcy + 30.0, rcz + 30.0)
            pd = vec3(rcx + 30.0, rcy + 30.0 + sft, rcz + 30.0)
            n2 = 2.0 * (noise3(pc * 3.0) * 0.5 + 0.5) * (noise3(pd * 6.0) * 0.5 + 0.5) \
                * (noise3(pc * 12.0) * 0.5 + 0.5) * (noise3(pd * 24.0) * 0.5 + 0.5) \
                * (noise3(pc * 48.0) * 0.5 + 0.5) * (noise3(pd * 92.0) * 0.5 + 0.5)
```

替换为（新增的扭曲/剪切/各向异性在 0/1 默认下均为恒等变换）：

```python
        if cov0 >= 0.01:
            ang = ti.atan2(y, x)          # 切口在 -x（黑洞背后），避开正前方那条接缝
            rcx = (dc * 1.5 + 0.55) * 0.95
            # 方位各向异性：DISK_FILAMENT>1 时沿切向拉长结构（除以比值=降方位频率）。=1 时恒等
            rcy = (ang * 1.5) * 0.95 / DISK_FILAMENT
            rcz = (dd * 1.5) * 0.95
            # 较差自转：内快外慢的相位增量。DISK_SHEAR=0 时退化为原版均匀 sft
            shear = (DISK_RIN / ti.max(rho_cyl, DISK_RIN)) ** 1.5 - 1.0
            sft = tparam * TIME_SPEED * (1.0 + DISK_SHEAR * shear)
            # 域扭曲：低频噪声向量扰动采样坐标。DISK_WARP=0 时 w=0，恒等
            base = vec3(rcx, rcy, rcz)
            w = DISK_WARP * vec3(noise3(base * 0.5),
                                 noise3(base * 0.5 + vec3(11.0, 17.0, 23.0)),
                                 noise3(base * 0.5 + vec3(5.0, 9.0, 1.0)))
            pa = vec3(rcx, rcy + sft, rcz) + w
            pb = vec3(rcx, rcy, rcz) + w
            n1 = (noise3(pa * 3.0) * 0.5 + 0.5) * (noise3(pb * 6.0) * 0.5 + 0.5) \
                * (noise3(pa * 12.0) * 0.5 + 0.5) * (noise3(pb * 24.0) * 0.5 + 0.5)
            if ti.static(DISK_OCTAVE_BOOST_EFF >= 1):
                n1 *= (noise3(pa * 48.0) * 0.5 + 0.5)
            if ti.static(DISK_OCTAVE_BOOST_EFF >= 2):
                n1 *= (noise3(pb * 96.0) * 0.5 + 0.5)
            pc = vec3(rcx + 30.0, rcy + 30.0, rcz + 30.0) + w
            pd = vec3(rcx + 30.0, rcy + 30.0 + sft, rcz + 30.0) + w
            n2 = 2.0 * (noise3(pc * 3.0) * 0.5 + 0.5) * (noise3(pd * 6.0) * 0.5 + 0.5) \
                * (noise3(pc * 12.0) * 0.5 + 0.5) * (noise3(pd * 24.0) * 0.5 + 0.5) \
                * (noise3(pc * 48.0) * 0.5 + 0.5) * (noise3(pd * 92.0) * 0.5 + 0.5)
```

> 注：`rho_cyl` 在本函数上文已定义（`rho_cyl = ti.sqrt(x*x + y*y)`），`DISK_RIN` 已在原 import 列表中。

- [ ] **Step 4: 回归测试应仍通过（默认值=原版）**

Run: `python -m pytest tests/test_regression.py::test_default_render_matches_baseline -v`
Expected: PASS（DISK_WARP=DISK_SHEAR=0、DISK_FILAMENT=1、boost=0 → 恒等）。

- [ ] **Step 5: 追加 A 的 effect 用例到 `tests/test_regression.py`**

```python
def test_disk_warp_changes_output():
    base = np.load(BASELINE)
    img = _probe_subprocess(["DISK_WARP=0.8", "DISK_SHEAR=1.2"])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"湍流旋钮未生效，max diff 仅 {diff.max():.4g}"
```

- [ ] **Step 6: 运行 effect 测试，应通过**

Run: `python -m pytest tests/test_regression.py::test_disk_warp_changes_output -v`
Expected: PASS（子进程以 warp/shear 开启重渲，画面应明显不同）。

- [ ] **Step 7: 肉眼复核**

Run: `python tools/probe.py --out probe_A.npy --set DISK_WARP=0.8 --set DISK_SHEAR=1.2 --set DISK_FILAMENT=2.0`
打开 `probe_A.png`，确认盘呈现拖尾螺旋丝状结构（而非原块状）。确认后删除 `probe_A.npy/png`。

- [ ] **Step 8: 提交**

```bash
git add config.py shading.py tests/test_regression.py
git commit -m "feat(A): 盘体丝状湍流（域扭曲+较差自转剪切+各向异性+octave）"
```

---

## Task 3: 工作线 B —— 光子环 + 内缘分层

积分循环记录最近接近半径 `r_min`，循环后叠加刀锋光子环；内缘新增 halo/veil 两层（默认 0，回归安全）。

**Files:**
- Modify: `config.py`（B 旋钮 + 派生 `PHOTONRING_ON_EFF` 非必需，直接用 `ti.static(PHOTONRING_ON==1)`）
- Modify: `render.py:55-93`（积分循环 + 循环后光子环）
- Modify: `shading.py`（`gas_disc_sample` 内缘 halo/veil）
- Modify: `tests/test_regression.py`（追加 B effect 用例）

- [ ] **Step 1: config 加 B 旋钮**

在"吸积盘"段 `INNER_GLOW` 行附近插入：

```python
# ---- B 光子环 / 内缘分层（默认关闭/原版）----
PHOTONRING_ON = 0            # 0=关（回归安全）；1=叠加刀锋光子环
PHOTONRING_STRENGTH = 1.0    # 光子环亮度
PHOTONRING_WIDTH = 0.15      # 光子环径向宽度（高斯 sigma，单位 M）
PHOTONRING_TINT = (1.0, 0.9, 0.75)   # 光子环色染
PHOTONRING_RADIUS = -1.0     # 光子球半径；<=0 则按 SPIN 自动算（顺行赤道光子轨道）
INNER_HALO = 0.0             # 内缘中晕强度（0=原版无此层）
INNER_VEIL = 0.0             # 内缘宽幅泛光强度（0=原版无此层）
```

- [ ] **Step 2: render.py import 新名字**

`render.py` 顶部 `from config import (...)` 追加：`PHOTONRING_ON, PHOTONRING_STRENGTH, PHOTONRING_WIDTH, PHOTONRING_TINT, PHOTONRING_RADIUS`。

- [ ] **Step 3: render kernel 维护 r_min 并在循环后叠加光子环**

在 `render.py` 的 per-pixel 循环初始化处（`h_step = INT_STEP` 行附近，循环 `for step in range(MAX_STEPS):` 之前）加：

```python
        r_min = R_SKY
```

在循环体内、算出 `r_curr = rFromCoords(pos)` 之后紧接一行：

```python
            r_min = ti.min(r_min, r_curr)
```

在 `for step` 循环**结束之后**、`fields.samp_acc[i, j] += col` 之前插入：

```python
        if ti.static(PHOTONRING_ON == 1):
            r_ph = PHOTONRING_RADIUS
            if r_ph <= 0.0:
                r_ph = 2.0 * (1.0 + ti.cos(2.0 / 3.0 * ti.acos(-SPIN)))  # 顺行赤道光子轨道半径
            dr = (r_min - r_ph) / PHOTONRING_WIDTH
            ring = ti.exp(-dr * dr) * PHOTONRING_STRENGTH
            col += (1.0 - alpha) * ring * vec3(PHOTONRING_TINT[0],
                                               PHOTONRING_TINT[1], PHOTONRING_TINT[2])
```

> `R_SKY`、`SPIN` 已在 `render.py` 的 import 中；`alpha`、`col` 是循环外已声明的累加变量。

- [ ] **Step 4: shading.py 内缘 halo/veil 两层**

`shading.py` 顶部 import 追加：`INNER_HALO, INNER_VEIL`。

定位 `gas_disc_sample` 中这段（约 122-126 行）：

```python
        bcol = vec3(1.0, 1.0, 1.0) * ti.pow(bloom_f, 1.5)
        bcol *= lerp(vec3(1.7, 1.1, 1.0), vec3(0.5, 0.6, 1.0), rg * rg)
        bcol *= lerp(vec3(1.7, 0.5, 0.1), vec3(1.0, 1.0, 1.0), ti.sqrt(rg))
        dust = lerp(dust, bcol * 150.0 * INNER_GLOW, ti.min(ti.max(1.0 - cov0, 0.0), 1.0))
```

替换为（在原 lerp 之后追加两层加性辉光，默认 0 → 恒等）：

```python
        bcol = vec3(1.0, 1.0, 1.0) * ti.pow(bloom_f, 1.5)
        bcol *= lerp(vec3(1.7, 1.1, 1.0), vec3(0.5, 0.6, 1.0), rg * rg)
        bcol *= lerp(vec3(1.7, 0.5, 0.1), vec3(1.0, 1.0, 1.0), ti.sqrt(rg))
        dust = lerp(dust, bcol * 150.0 * INNER_GLOW, ti.min(ti.max(1.0 - cov0, 0.0), 1.0))
        # 内缘分层：中晕（较宽径向权重）+ 宽幅泛光（更宽）。默认强度 0，恒等回归
        dust += bcol * 30.0 * INNER_HALO * pcurve(rg, 2.0, 1.5)
        dust += bcol * 8.0 * INNER_VEIL * pcurve(rg, 1.0, 2.0)
```

> `pcurve`、`rg`、`bcol`、`cov0`、`INNER_GLOW` 均在本函数作用域/模块内已有。

- [ ] **Step 5: 回归测试应仍通过**

Run: `python -m pytest tests/test_regression.py::test_default_render_matches_baseline -v`
Expected: PASS（PHOTONRING_ON=0 → `ti.static` 整块不编译；INNER_HALO=INNER_VEIL=0 → 加 0）。

- [ ] **Step 6: 追加 B effect 用例**

```python
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
```

- [ ] **Step 7: 运行 B effect 测试**

Run: `python -m pytest tests/test_regression.py -k "photon_ring or inner_layers" -v`
Expected: 两条 PASS。

- [ ] **Step 8: 肉眼复核**

Run: `python tools/probe.py --out probe_B.npy --set PHOTONRING_ON=1 --set PHOTONRING_STRENGTH=3.0 --set INNER_HALO=2.0`
打开 `probe_B.png`，确认黑洞剪影外出现细亮环、内缘有结构化层次。确认后删除临时文件。

- [ ] **Step 9: 提交**

```bash
git add config.py render.py shading.py tests/test_regression.py
git commit -m "feat(B): 光子环 + 内缘多层辉光"
```

---

## Task 4: 工作线 C —— 图像底图星空

下载等距柱状银河星图 → fields 加载 → `sky_color` 采样。`STARMAP_PATH=None` 时回退原程序化（回归安全）。

**Files:**
- Create: `assets/starmap_2020_4k.tif`、`assets/STARMAP_SOURCE.md`
- Modify: `config.py`（C 旋钮 + 派生 `STARMAP_ON`、`STARMAP_W/H`）
- Modify: `fields.py`（`starmap_tex` + `load_starmap_tex`）
- Modify: `shading.py`（`sky_color` 拆 `_proc_sky` + 星图采样）
- Modify: `main.py`（加载星图）
- Modify: `tests/test_regression.py`（C effect 用例）

- [ ] **Step 1: 下载星图素材**

来源：NASA SVS "Deep Star Maps 2020"（公有领域，等距柱状天球图）。主页：`https://svs.gsfc.nasa.gov/4851`。下载 4K 版（4096×2048）TIFF 到 `assets/`。

PowerShell（先确认主页上 4k 文件的实际链接；若下方直链失效，从主页"Download"区取 4096×2048 的 tif/png）：

```powershell
Invoke-WebRequest -Uri "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_4k.tif" -OutFile "assets/starmap_2020_4k.tif"
```

校验：文件存在且分辨率约 4096×2048（宽高比 2:1）。失败则改用 ESO Milky Way 全景（`https://www.eso.org/public/images/eso0932a/`，CC BY 4.0，需在下方记录署名）。

- [ ] **Step 2: 记录来源/授权 `assets/STARMAP_SOURCE.md`**

```markdown
# 星图素材来源

文件：starmap_2020_4k.tif
来源：NASA/Goddard SVS — Deep Star Maps 2020 (https://svs.gsfc.nasa.gov/4851)
授权：公有领域（NASA SVS media usage guidelines）
说明：等距柱状（equirectangular）天球全景，用作渲染器逃逸光线背景采样。
```

- [ ] **Step 3: config 加 C 旋钮 + 派生**

"动画"段之前、`R_SKY` 附近插入：

```python
# ---- C 图像底图星空 ----
STARMAP_PATH = _rel("assets/starmap_2020_4k.tif")  # None=回退纯程序化星空（回归安全）
STARMAP_RES = (4096, 2048)   # 星图采样分辨率（W,H，2:1）。显存=W*H*3*4 字节≈100MB
STARMAP_GAIN = 1.0           # 底图亮度增益
STARMAP_ROT = 0.0            # 方位旋转（绕极轴，单位"圈"，对准银河）
STARMAP_FLIP = 0             # 1=上下翻转 v（朝向修正）
STARMAP_PROC_MIX = 0.3       # 程序化近景亮星叠加权重（0=纯图）
```

文件末尾派生段追加：

```python
# C：星图是否启用（编译期常量，被 ti.static 用）+ 尺寸常量
STARMAP_ON = 1 if STARMAP_PATH else 0
STARMAP_W, STARMAP_H = STARMAP_RES
```

- [ ] **Step 4: fields.py 加 starmap_tex + loader**

`fields.py` 顶部 import 追加：`STARMAP_W, STARMAP_H, STARMAP_ON`。模块级声明区加 `starmap_tex = None`。

在 `build()` 里、`organic_tex` 分配之后追加：

```python
    global starmap_tex
    if STARMAP_ON == 1:
        starmap_tex = ti.Vector.field(3, dtype=ti.f32, shape=(STARMAP_W, STARMAP_H))
    else:
        starmap_tex = ti.Vector.field(3, dtype=ti.f32, shape=(1, 1))
```

> 即使关闭也分配 1×1 占位，避免 kernel 引用未分配 field。`build()` 的 `global` 声明行需把 `starmap_tex` 加进去。

文件末尾追加 loader（照搬 `load_organic_tex` 的 imread/降采样/转置）：

```python
def load_starmap_tex(path=None):
    """填充等距柱状星图（W×H，RGB）。无 path 则填黑。"""
    if path:
        img = ti.tools.imread(path).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img[..., :3]
        h, w = img.shape[0], img.shape[1]
        yi = (np.arange(STARMAP_H) * h // STARMAP_H) % h
        xi = (np.arange(STARMAP_W) * w // STARMAP_W) % w
        grid = img[np.ix_(yi, xi)]
        grid = np.ascontiguousarray(np.transpose(grid, (1, 0, 2)), dtype=np.float32)
    else:
        grid = np.zeros((STARMAP_W, STARMAP_H, 3), dtype=np.float32)
    starmap_tex.from_numpy(grid)
```

- [ ] **Step 5: shading.py 拆 `_proc_sky` 并接入星图**

`shading.py` 顶部 import 追加：`STARMAP_ON, STARMAP_W, STARMAP_H, STARMAP_GAIN, STARMAP_ROT, STARMAP_FLIP, STARMAP_PROC_MIX`。

把现有 `sky_color` 整体改名为 `_proc_sky`（函数体不变），然后新增星图采样与新的 `sky_color`：

```python
@ti.func
def sample_starmap(u: ti.f32, v: ti.f32) -> vec3:
    """等距柱状双线性采样：u 方位 wrap，v 极向钳位。"""
    gx = u * STARMAP_W
    gy = v * STARMAP_H
    i0 = int(ti.floor(gx))
    j0 = int(ti.floor(gy))
    fx = gx - i0
    fy = gy - j0
    i0 = i0 % STARMAP_W
    i1 = (i0 + 1) % STARMAP_W
    j0 = ti.min(ti.max(j0, 0), STARMAP_H - 1)
    j1 = ti.min(j0 + 1, STARMAP_H - 1)
    a = fields.starmap_tex[i0, j0]
    b = fields.starmap_tex[i1, j0]
    c = fields.starmap_tex[i0, j1]
    d = fields.starmap_tex[i1, j1]
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy)


@ti.func
def sky_color(theta: ti.f32, phi: ti.f32) -> vec3:
    col = vec3(0.0, 0.0, 0.0)
    if ti.static(STARMAP_ON == 1):
        u = ti.math.fract(phi / (2.0 * math.pi) + STARMAP_ROT)
        v = ti.min(ti.max(theta / math.pi, 0.0), 1.0)
        if ti.static(STARMAP_FLIP == 1):
            v = 1.0 - v
        col = sample_starmap(u, v) * STARMAP_GAIN
        col += STARMAP_PROC_MIX * _proc_sky(theta, phi)
    else:
        col = _proc_sky(theta, phi)
    return col
```

> `render.py` 已调用 `sky_color(th, ph)`，签名不变。引力透镜自动生效（按逃逸方向采样）。

- [ ] **Step 6: main.py 加载星图**

`main.py` 的 `main()` 中，`fields.load_organic_tex(...)` 之后追加：

```python
    from config import STARMAP_PATH
    if STARMAP_PATH:
        fields.load_starmap_tex(STARMAP_PATH)
```

- [ ] **Step 7: 回归——基线需在 STARMAP_PATH=None 下比较**

基线是在无星图时录的；接入星图后默认 `STARMAP_PATH` 指向素材，会改变星空 → 回归用例若直接跑会因星空变化而"失败"。**处理方式**：回归用例验证的是"新旋钮取原版值"，对 C 而言原版值是 `STARMAP_PATH=None`。修改回归用例让探针在 None 下比对，并新增"星图开启会改变输出"的 effect 用例。

把 `tests/test_regression.py` 的 `test_default_render_matches_baseline` 改为强制关闭星图：

```python
def test_default_render_matches_baseline():
    from tools.probe import render_probe
    img = render_probe(overrides={"STARMAP_PATH": None, "STARMAP_ON": 0})
    assert os.path.exists(BASELINE), "基线缺失：先跑 Task 1"
    base = np.load(BASELINE)
    assert img.shape == base.shape
    diff = np.abs(img - base)
    assert diff.max() < 2e-3, f"max diff {diff.max():.4g} 超过容差（回归被破坏）"
```

> `STARMAP_ON` 是 `ti.static` 编译期常量，必须连同 `STARMAP_PATH` 一起覆盖到 shading 模块，`render_probe` 的 `_apply` 已对所有模块 setattr，生效。

- [ ] **Step 8: 回归测试应通过**

Run: `python -m pytest tests/test_regression.py::test_default_render_matches_baseline -v`
Expected: PASS。

- [ ] **Step 9: 加 C effect 用例**

```python
def test_starmap_changes_sky():
    base = np.load(BASELINE)
    # 子进程默认 STARMAP_PATH 已指向素材 → 星空应与程序化基线不同
    img = _probe_subprocess([])
    diff = np.abs(img - base)
    assert diff.max() > 0.02, f"星图底图未改变星空，max diff 仅 {diff.max():.4g}"
```

> 注意：此用例依赖 `assets/` 中已下载星图。子进程探针默认加载 `config.STARMAP_PATH`。

- [ ] **Step 10: 运行 C effect 测试**

Run: `python -m pytest tests/test_regression.py::test_starmap_changes_sky -v`
Expected: PASS。

- [ ] **Step 11: 肉眼复核**

Run: `python tools/probe.py --out probe_C.npy`
打开 `probe_C.png`，确认背景为真实银河星图、且黑洞附近星空被透镜扭曲。确认后删临时文件。

- [ ] **Step 12: 提交**

```bash
git add config.py fields.py shading.py main.py assets/STARMAP_SOURCE.md tests/test_regression.py
git add assets/starmap_2020_4k.tif
git commit -m "feat(C): 图像底图星空（等距柱状银河贴图 + 透镜采样）"
```

> 若星图文件较大，确认未被 `.gitignore` 排除；如仓库不宜放二进制大图，改为在 `STARMAP_SOURCE.md` 记录下载方式并把图加入 `.gitignore`（此时 effect 用例改为存在性跳过）。执行时按实际仓库策略决定，并在提交信息注明。

---

## Task 5: 工作线 D —— 后期电影化

tonemap 选择 + anamorphic 拉丝 + vignette/grain/chroma。全部默认 0/原版曲线（回归安全）。

**Files:**
- Modify: `config.py`（D 旋钮）
- Modify: `postprocess.py`（tonemap 分支 + anamorphic kernel + composite 收尾）
- Modify: `tests/test_regression.py`（D effect 用例）

- [ ] **Step 1: config 加 D 旋钮**

"后处理"段末尾插入：

```python
# ---- D 后期电影化（默认全取原版/关闭）----
TONEMAP = 0                  # 0=现有曲线 / 1=ACES / 2=AgX 近似
ANAMORPHIC_ON = 0            # 1=横向镜头拉丝
ANAMORPHIC_STRENGTH = 0.5    # 拉丝强度
ANAMORPHIC_TINT = (0.4, 0.6, 1.0)  # 拉丝色染（经典冷蓝）
VIGNETTE_STRENGTH = 0.0      # 暗角强度（0=关）
GRAIN_STRENGTH = 0.0         # 胶片颗粒幅度（0=关）
CHROMA_AB_STRENGTH = 0.0     # 边缘色散强度（0=关）
```

- [ ] **Step 2: postprocess.py import 新名字**

`postprocess.py` 顶部 import 追加：`TONEMAP, ANAMORPHIC_ON, ANAMORPHIC_STRENGTH, ANAMORPHIC_TINT, VIGNETTE_STRENGTH, GRAIN_STRENGTH, CHROMA_AB_STRENGTH`，以及 `import math`（grain 哈希用，若未导入）。

- [ ] **Step 3: 加 tonemap 辅助函数（ACES / AgX 近似）**

`postprocess.py` 的 `luma` 之后追加：

```python
@ti.func
def tonemap_aces(c: vec3) -> vec3:
    # Narkowicz ACES filmic 近似
    a, b, cc, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ti.math.clamp((c * (a * c + b)) / (c * (cc * c + d) + e), 0.0, 1.0)


@ti.func
def tonemap_agx(c: vec3) -> vec3:
    # AgX 极简近似：log 编码 + S 形 + 还原（够电影向，非精确）
    x = ti.max(c, vec3(0.0, 0.0, 0.0))
    x = x / (x + vec3(1.0, 1.0, 1.0))
    x = ti.math.pow(x, vec3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
    return ti.math.clamp(x * x * (vec3(3.0, 3.0, 3.0) - 2.0 * x), 0.0, 1.0)
```

- [ ] **Step 4: 改 `k_composite` —— tonemap 分支 + 收尾质感**

把 `k_composite` 整体替换为：

```python
@ti.func
def hash21p(p: vec2) -> ti.f32:
    return ti.math.fract(ti.sin(p.dot(vec2(127.1, 311.7))) * 43758.5453)


@ti.kernel
def k_composite(strength: ti.f32, post: ti.f32):
    nx = fields.pixels.shape[0]
    ny = fields.pixels.shape[1]
    for i, j in fields.pixels:
        # 边缘色散：CHROMA>0 时 R/B 通道按到中心的径向偏移分别采样
        cx = nx * 0.5
        cy = ny * 0.5
        rx = (i - cx) / nx
        ry = (j - cy) / ny
        r2 = rx * rx + ry * ry
        base = fields.pixels[i, j] + fields.bloom_acc[i, j] * strength
        c = base
        if ti.static(True):  # CHROMA 分支（默认强度 0 → 偏移 0，恒等）
            off = CHROMA_AB_STRENGTH * r2
            pr = bsample(fields.pixels, i - off * (i - cx), j - off * (j - cy)) \
                + bsample(fields.bloom_acc, i - off * (i - cx), j - off * (j - cy)) * strength
            pb = bsample(fields.pixels, i + off * (i - cx), j + off * (j - cy)) \
                + bsample(fields.bloom_acc, i + off * (i - cx), j + off * (j - cy)) * strength
            c = vec3(pr[0], base[1], pb[2])
        c = c * post
        # tonemap 分支
        if ti.static(TONEMAP == 1):
            c = tonemap_aces(c)
        elif ti.static(TONEMAP == 2):
            c = tonemap_agx(c)
        else:
            c = ti.math.pow(c, vec3(1.5, 1.5, 1.5))
            c = c / (vec3(1.0, 1.0, 1.0) + c)
            c = ti.math.pow(c, vec3(1.0 / 1.5, 1.0 / 1.5, 1.0 / 1.5))
            cs = c * c * (vec3(3.0, 3.0, 3.0) - 2.0 * c)
            c = lerp(c, cs, CONTRAST)
            c = ti.math.pow(c, vec3(1.3, 1.20, 1.0))
            c = ti.math.clamp(c * 1.01, 0.0, 1.0)
            c = ti.math.pow(c, vec3(0.7 / 2.2, 0.7 / 2.2, 0.7 / 2.2))
        # 暗角（默认 0 恒等）
        c *= 1.0 - VIGNETTE_STRENGTH * r2 * 4.0
        # 胶片颗粒（默认 0 恒等）
        gn = (hash21p(vec2(ti.cast(i, ti.f32), ti.cast(j, ti.f32))) - 0.5)
        c += GRAIN_STRENGTH * gn
        c = ti.math.clamp(c, 0.0, 1.0)
        fields.disp[i, j] = c * vec3(OUTPUT_TINT[0], OUTPUT_TINT[1], OUTPUT_TINT[2])
```

> 用到 `bsample`（同文件已定义）、`vec2`/`vec3`/`lerp`/`CONTRAST`/`OUTPUT_TINT`（已 import）。`vec2` 需在 import 中——`postprocess.py` 顶部 `from config import (vec3, ...)`，追加 `vec2`。
>
> 回归安全性：CHROMA=0 → off=0 → `bsample(...,i,j)` 等于原值（`pr[0]`/`pb[2]` 与 base 同），但浮点上 bsample 在整数坐标应精确返回原值（fx=fy=0）；VIGNETTE=0、GRAIN=0 → 不变；TONEMAP=0 → 原曲线逐行复刻。若回归因 bsample 浮点产生极小偏差，容差 2e-3 可吸收。

- [ ] **Step 5: 加 anamorphic 拉丝 kernel 与接入**

`postprocess.py` 加横向拉丝 kernel（从最宽的 bloom 级做横向 1D 宽模糊，累加进 `bloom_acc`）：

```python
@ti.kernel
def k_anamorphic(src: ti.template(), strength: ti.f32,
                 tr: ti.f32, tg: ti.f32, tb: ti.f32):
    """横向 1D 宽模糊镜头拉丝，结果加权累加到 bloom_acc（与主 bloom 同分辨率上采样）。"""
    nx = fields.bloom_acc.shape[0]
    ny = fields.bloom_acc.shape[1]
    sx = src.shape[0]
    sy = src.shape[1]
    for i, j in fields.bloom_acc:
        u = (i + 0.5) / nx * sx
        v = (j + 0.5) / ny * sy
        acc = vec3(0.0, 0.0, 0.0)
        wsum = 0.0
        for k in range(-12, 13):
            w = ti.exp(-(k * k) / 40.0)
            acc += bsample(src, u + k * 1.5, v) * w
            wsum += w
        acc = acc / wsum
        fields.bloom_acc[i, j] += acc * strength * vec3(tr, tg, tb)
```

把 `_bloom_dual` 之后、`postprocess()` 中对 `k_composite` 的调用之前接入：

```python
def postprocess():
    k_expose(EXPOSURE_PRE)
    _bloom_dual()
    if ANAMORPHIC_ON == 1:
        k_anamorphic(fields.bloom_lv[2], ANAMORPHIC_STRENGTH,
                     ANAMORPHIC_TINT[0], ANAMORPHIC_TINT[1], ANAMORPHIC_TINT[2])
    k_composite(BLOOM_STRENGTH, EXPOSURE_POST)
```

> `ANAMORPHIC_ON==1` 是普通 Python 分支（`postprocess()` 是普通函数），关闭时不调用，回归安全。`fields.bloom_lv[2]` 为中等模糊级（确保 `BLOOM_LEVELS>=3`，当前为 8）。

- [ ] **Step 6: 回归测试应通过**

Run: `python -m pytest tests/test_regression.py::test_default_render_matches_baseline -v`
Expected: PASS。

> 若因 CHROMA 的 bsample 路径产生 >2e-3 偏差，将 Step 4 的 CHROMA 分支包到 `if ti.static(CHROMA_AB_STRENGTH > 0.0):` 内，关闭时彻底走 `c = base` 原路径。

- [ ] **Step 7: 加 D effect 用例**

```python
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
```

- [ ] **Step 8: 运行 D effect 测试**

Run: `python -m pytest tests/test_regression.py -k "tonemap or anamorphic" -v`
Expected: 两条 PASS。

- [ ] **Step 9: 肉眼复核**

Run: `python tools/probe.py --out probe_D.npy --set TONEMAP=2 --set ANAMORPHIC_ON=1 --set ANAMORPHIC_STRENGTH=1.2 --set VIGNETTE_STRENGTH=0.4`
打开 `probe_D.png`，确认高光滚降更柔、亮区有横向蓝色拉丝、四角轻微压暗。确认后删临时文件。

- [ ] **Step 10: 提交**

```bash
git add config.py postprocess.py tests/test_regression.py
git commit -m "feat(D): 后期电影化（ACES/AgX + anamorphic 拉丝 + vignette/grain/chroma）"
```

---

## Task 6: 全开 4K 对比图 + 全测试

**Files:**
- 无新增（产出对比图，不提交大图）

- [ ] **Step 1: 跑全部测试**

Run: `python -m pytest tests/test_regression.py -v`
Expected: 回归 + 全部 effect 用例 PASS。

- [ ] **Step 2: 选一组"全开"推荐值出探针对比**

Run（一次原版、一次全开）：

```powershell
python tools/probe.py --out probe_off.npy
python tools/probe.py --out probe_on.npy --set DISK_WARP=0.7 --set DISK_SHEAR=1.0 --set DISK_FILAMENT=1.8 --set DISK_OCTAVE_BOOST=1 --set PHOTONRING_ON=1 --set PHOTONRING_STRENGTH=2.5 --set INNER_HALO=1.5 --set INNER_VEIL=1.0 --set STARMAP_PROC_MIX=0.3 --set TONEMAP=2 --set ANAMORPHIC_ON=1 --set ANAMORPHIC_STRENGTH=1.0 --set VIGNETTE_STRENGTH=0.35
```

> `DISK_OCTAVE_BOOST` 通过 `--set` 覆盖到模块时，注意它经由 `DISK_OCTAVE_BOOST_EFF` 派生且被 `ti.static` 使用；probe 子进程为离线档（`INTERACTIVE=False`），需同时覆盖 `DISK_OCTAVE_BOOST_EFF`。把上面命令再加 `--set DISK_OCTAVE_BOOST_EFF=1`，确保该 octave 生效（`_apply` 会写到 shading 模块）。

- [ ] **Step 3: 肉眼对比 `probe_off.png` vs `probe_on.png`**

确认四条线叠加后整体电影观感明显提升、无明显瑕疵（盘不过曝、光子环不锯齿、星图无接缝）。

- [ ] **Step 4: 把推荐"全开"值写入 config 默认（可选，按口味微调）**

若满意，将 Step 2 的推荐值更新到 `config.py` 各旋钮默认（离线出图直接生效；GUI 仍自动降级）。否则保留默认关闭、由用户按需开。

- [ ] **Step 5: 出一张正式 4K 静帧确认**

把 `config.py` 的 `INTERACTIVE=False`、`ANIMATE=False`，按口味设好旋钮，Run: `python main.py`。检查 `interstellar_ks_4k.png`。

- [ ] **Step 6: 清理临时探针文件并最终提交**

```bash
git add config.py
git commit -m "chore: 电影化增强推荐参数默认值 + 收尾"
```

---

## Self-Review 记录

- **Spec 覆盖**：A→Task2，B→Task3，C→Task4，D→Task5，回归保证→Task0/1 + 各任务回归步骤，GUI 降级→`DISK_OCTAVE_BOOST_EFF`（A）/ `ti.static` 门控（B/C）/ Python 分支（D anamorphic）。验收"全开对比图"→Task6。无遗漏。
- **占位符扫描**：无 TBD/TODO；每个代码步骤含完整可应用代码。唯一外部不确定项为星图直链 URL（Task4 Step1 已给主页 + 校验 + ESO 备选）。
- **类型/命名一致**：`render_probe(overrides)`、`_probe_subprocess(sets)`、`load_starmap_tex`、`sample_starmap`、`_proc_sky`、`tonemap_aces/agx`、`k_anamorphic`、派生 `STARMAP_ON/STARMAP_W/STARMAP_H/DISK_OCTAVE_BOOST_EFF` 在定义与引用处一致。
- **已知风险**：(1) GPU 浮点非确定性可能使回归差异接近容差——容差 2e-3 留有余量，必要时调大或固定到特定 GPU；(2) D 的 CHROMA bsample 路径若引入微偏差，Task5 Step6 给了 `ti.static` 彻底短路的回退；(3) 星图大文件入库策略在 Task4 Step12 给了 .gitignore 备选。
