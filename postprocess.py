"""
后处理：Bloom（Jimenez/COD dual-filter）+ 两段曝光 + filmic 色调映射。

bloom 的下/上采样 kernel 用 ti.template() 参数收发 field，调用处（_bloom_dual）传 fields.xxx。
k_composite / k_expose 直接读写 fields.pixels / bloom_acc / disp。
"""
import taichi as ti

from config import (vec3, BLOOM_LEVELS, BLOOM_STRENGTH,
                    EXPOSURE_PRE, EXPOSURE_POST, CONTRAST, HIGHLIGHT_SOFT, OUTPUT_TINT)
from common import lerp
import fields


@ti.func
def luma(c: vec3) -> ti.f32:
    return c.dot(vec3(0.2126, 0.7152, 0.0722))


@ti.func
def bsample(src: ti.template(), x: ti.f32, y: ti.f32) -> vec3:
    """双线性采样 src 于纹素坐标 (x,y)，边缘钳位。"""
    nx = src.shape[0]
    ny = src.shape[1]
    xc = ti.min(ti.max(x, 0.0), nx - 1.0)
    yc = ti.min(ti.max(y, 0.0), ny - 1.0)
    i0 = int(ti.floor(xc))
    j0 = int(ti.floor(yc))
    i1 = ti.min(i0 + 1, nx - 1)
    j1 = ti.min(j0 + 1, ny - 1)
    fx = xc - i0
    fy = yc - j0
    return (src[i0, j0] * (1.0 - fx) + src[i1, j0] * fx) * (1.0 - fy) \
        + (src[i0, j1] * (1.0 - fx) + src[i1, j1] * fx) * fy


@ti.kernel
def k_down13(src: ti.template(), dst: ti.template(), karis: ti.template()):
    """Jimenez/COD 13-tap 下采样（karis=1 时首级做亮度加权抑制 firefly）。"""
    rx = src.shape[0] / dst.shape[0]
    ry = src.shape[1] / dst.shape[1]
    for i, j in dst:
        cx = (i + 0.5) * rx - 0.5
        cy = (j + 0.5) * ry - 0.5
        a = bsample(src, cx - 2, cy - 2)
        b = bsample(src, cx, cy - 2)
        c = bsample(src, cx + 2, cy - 2)
        d = bsample(src, cx - 2, cy)
        e = bsample(src, cx, cy)
        f = bsample(src, cx + 2, cy)
        g = bsample(src, cx - 2, cy + 2)
        h = bsample(src, cx, cy + 2)
        ii = bsample(src, cx + 2, cy + 2)
        jj = bsample(src, cx - 1, cy - 1)
        kk = bsample(src, cx + 1, cy - 1)
        ll = bsample(src, cx - 1, cy + 1)
        mm = bsample(src, cx + 1, cy + 1)
        res = vec3(0.0, 0.0, 0.0)
        if ti.static(karis == 1):
            g0 = (a + b + d + e) * 0.25
            g1 = (b + c + e + f) * 0.25
            g2 = (d + e + g + h) * 0.25
            g3 = (e + f + h + ii) * 0.25
            g4 = (jj + kk + ll + mm) * 0.25
            w0 = 1.0 / (1.0 + luma(g0))
            w1 = 1.0 / (1.0 + luma(g1))
            w2 = 1.0 / (1.0 + luma(g2))
            w3 = 1.0 / (1.0 + luma(g3))
            w4 = 1.0 / (1.0 + luma(g4))
            sw = w0 * 0.125 + w1 * 0.125 + w2 * 0.125 + w3 * 0.125 + w4 * 0.5
            res = (g0 * (w0 * 0.125) + g1 * (w1 * 0.125) + g2 * (w2 * 0.125)
                   + g3 * (w3 * 0.125) + g4 * (w4 * 0.5)) / sw
        else:
            res = e * 0.125 + (a + c + g + ii) * 0.03125 \
                + (b + d + f + h) * 0.0625 + (jj + kk + ll + mm) * 0.125
        dst[i, j] = res


@ti.kernel
def k_up_tent_add(src: ti.template(), dst: ti.template()):
    """3×3 tent 升采样并累加到更大一级（= COD 渐进上采样链）。"""
    rx = src.shape[0] / dst.shape[0]
    ry = src.shape[1] / dst.shape[1]
    for i, j in dst:
        sx = (i + 0.5) * rx - 0.5
        sy = (j + 0.5) * ry - 0.5
        a = bsample(src, sx - 1, sy + 1)
        b = bsample(src, sx, sy + 1)
        c = bsample(src, sx + 1, sy + 1)
        d = bsample(src, sx - 1, sy)
        e = bsample(src, sx, sy)
        f = bsample(src, sx + 1, sy)
        g = bsample(src, sx - 1, sy - 1)
        h = bsample(src, sx, sy - 1)
        ii = bsample(src, sx + 1, sy - 1)
        tent = (e * 4.0 + (b + d + f + h) * 2.0 + (a + c + g + ii) * 1.0) / 16.0
        dst[i, j] += tent


@ti.kernel
def k_clear(f: ti.template()):
    for i, j in f:
        f[i, j] = vec3(0.0, 0.0, 0.0)


@ti.kernel
def k_composite(strength: ti.f32, post: ti.f32):
    for i, j in fields.pixels:
        c = (fields.pixels[i, j] + fields.bloom_acc[i, j] * strength) * post   # = 原版 (color+bloom*0.08)*200
        c = ti.math.pow(c, vec3(1.5, 1.5, 1.5))
        c = c / (vec3(1.0, 1.0, 1.0) + c)
        c = ti.math.pow(c, vec3(1.0 / 1.5, 1.0 / 1.5, 1.0 / 1.5))
        cs = c * c * (vec3(3.0, 3.0, 3.0) - 2.0 * c)         # smoothstep S 曲线
        c = lerp(c, cs, CONTRAST)                            # 按 CONTRAST 调对比强度（<1 压缩动态范围）
        # 原版 lstSRS 的轻微冷色调（R/G 指数>1 → 相对压低红绿、偏冷）。
        # 与下方 OUTPUT_TINT 偏橙方向相反、互相抵消一部分；想更暖可把这行三个指数都设回 1.0。
        c = ti.math.pow(c, vec3(1.3, 1.20, 1.0))
        c = ti.math.clamp(c * 1.01, 0.0, 1.0)
        c = ti.math.pow(c, vec3(0.7 / 2.2, 0.7 / 2.2, 0.7 / 2.2))   # gamma（HDR→显示域）
        fields.disp[i, j] = c * vec3(OUTPUT_TINT[0], OUTPUT_TINT[1], OUTPUT_TINT[2])  # LDR 域整体染色（偏橙）


@ti.kernel
def k_expose(e: ti.f32):
    """×e 后高光处理：HIGHLIGHT_SOFT=0 硬 saturate（原版死白）；=1 软滚降 1-e^-x（高光保留层次）。"""
    for i, j in fields.pixels:
        x = ti.max(fields.pixels[i, j] * e, vec3(0.0, 0.0, 0.0))
        hard = ti.min(x, vec3(1.0, 1.0, 1.0))
        soft = vec3(1.0, 1.0, 1.0) - ti.math.exp(-x)
        fields.pixels[i, j] = lerp(hard, soft, HIGHLIGHT_SOFT)


def _bloom_dual():
    """Jimenez/COD dual-filter：13-tap 下采样(首级 Karis) + 3×3 tent 渐进上采样。"""
    k_down13(fields.pixels, fields.bloom_lv[0], 1)
    for i in range(1, BLOOM_LEVELS):
        k_down13(fields.bloom_lv[i - 1], fields.bloom_lv[i], 0)
    for i in range(BLOOM_LEVELS - 1, 0, -1):
        k_up_tent_add(fields.bloom_lv[i], fields.bloom_lv[i - 1])
    k_clear(fields.bloom_acc)
    k_up_tent_add(fields.bloom_lv[0], fields.bloom_acc)


def postprocess():
    k_expose(EXPOSURE_PRE)                 # ×PRE → 高光处理（= 原版 BufferA 输出）
    _bloom_dual()                          # Jimenez/COD dual-filter bloom
    k_composite(BLOOM_STRENGTH, EXPOSURE_POST)   # 加 bloom → ×POST → tonemap（= 原版 Image）
