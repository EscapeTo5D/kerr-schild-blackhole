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
    # 注意：每个 Python 进程只应调用一次（fields.build 会重分配 field，
    # Taichi runtime 已初始化后重复 build 不安全）。多组参数对比请走 _probe_subprocess（各自全新进程）。
    # 先降分辨率（务必在首次 render 编译前）
    for m in (config, fields, render):
        if hasattr(m, "RES_X"):
            m.RES_X, m.RES_Y = PROBE_RES
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
