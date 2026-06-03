"""跨模块共用的 Taichi 小工具。"""
import taichi as ti


@ti.func
def lerp(x, y, a):
    return x * (1.0 - a) + y * a
