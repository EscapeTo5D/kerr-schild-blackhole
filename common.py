"""跨模块共用的 Taichi 小工具。"""
import taichi as ti


@ti.func
def lerp(x, y, a):
    return x * (1.0 - a) + y * a


@ti.func
def hash21(p: ti.math.vec2) -> ti.f32:
    return ti.math.fract(ti.sin(p.dot(ti.math.vec2(127.1, 311.7))) * 43758.5453)
