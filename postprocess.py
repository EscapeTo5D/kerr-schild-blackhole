"""
后处理：Bloom（Jimenez/COD dual-filter）+ 两段曝光 + filmic 色调映射。

bloom 的下/上采样 kernel 用 ti.template() 参数收发 field，调用处（_bloom_dual）传 fields.xxx。
k_composite / k_expose 直接读写 fields.pixels / bloom_acc / disp。
"""
import taichi as ti

from config import (vec2, vec3, BLOOM_LEVELS, BLOOM_STRENGTH,
                    EXPOSURE_PRE, EXPOSURE_POST, CONTRAST, HIGHLIGHT_SOFT, OUTPUT_TINT,
                    TONEMAP, ANAMORPHIC_ON, ANAMORPHIC_STRENGTH, ANAMORPHIC_TINT,
                    VIGNETTE_STRENGTH, GRAIN_STRENGTH, CHROMA_AB_STRENGTH)
from common import lerp, hash21
import fields


@ti.func
def luma(c: vec3) -> ti.f32:
    return c.dot(vec3(0.2126, 0.7152, 0.0722))


@ti.func
def tonemap_aces(c: vec3) -> vec3:
    # Narkowicz ACES filmic 近似
    a, b, cc, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ti.math.clamp((c * (a * c + b)) / (c * (cc * c + d) + e), 0.0, 1.0)


@ti.func
def tonemap_agx(c: vec3) -> vec3:
    # AgX 极简近似：压缩 + S 形 + gamma（电影向高光滚降，非精确）
    x = ti.max(c, vec3(0.0, 0.0, 0.0))
    x = x / (x + vec3(1.0, 1.0, 1.0))
    x = ti.math.pow(x, vec3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2))
    return ti.math.clamp(x * x * (vec3(3.0, 3.0, 3.0) - 2.0 * x), 0.0, 1.0)


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
    nx = fields.pixels.shape[0]
    ny = fields.pixels.shape[1]
    for i, j in fields.pixels:
        cx = nx * 0.5
        cy = ny * 0.5
        rx = (i - cx) / nx
        ry = (j - cy) / ny
        r2 = rx * rx + ry * ry
        base = fields.pixels[i, j] + fields.bloom_acc[i, j] * strength
        c = base
        # 边缘色散：开启后每像素 +4 次双线性采样（实时模式慎用）；CHROMA=0 时整块不编译，bit-exact
        # 边缘色散：仅当 CHROMA>0 才编译该分支，默认 c=base 严格原路径（bit-exact 回归）
        if ti.static(CHROMA_AB_STRENGTH > 0.0):
            off = CHROMA_AB_STRENGTH * r2
            fi = ti.cast(i, ti.f32)
            fj = ti.cast(j, ti.f32)
            pr = bsample(fields.pixels, fi - off * (i - cx), fj - off * (j - cy)) \
                + bsample(fields.bloom_acc, fi - off * (i - cx), fj - off * (j - cy)) * strength
            pb = bsample(fields.pixels, fi + off * (i - cx), fj + off * (j - cy)) \
                + bsample(fields.bloom_acc, fi + off * (i - cx), fj + off * (j - cy)) * strength
            c = vec3(pr[0], base[1], pb[2])
        c = c * post
        # tonemap 分支
        if ti.static(TONEMAP == 1):
            # 注：CONTRAST 旋钮不作用于 ACES/AgX（自带对比度）
            c = tonemap_aces(c)
        elif ti.static(TONEMAP == 2):
            # 注：CONTRAST 旋钮不作用于 ACES/AgX（自带对比度）
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
        # 暗角：×4.0 使 STRENGTH=0.5 时角落(r2≈0.5)恰全暗；>0.5 靠末尾 clamp 兜底。默认 0 → ×1.0 恒等
        c *= 1.0 - VIGNETTE_STRENGTH * r2 * 4.0
        # 胶片颗粒：ti.static 守卫使 GRAIN=0 时整块不编译，省每像素一次 hash；默认恒等
        if ti.static(GRAIN_STRENGTH > 0.0):
            gn = hash21(vec2(ti.cast(i, ti.f32), ti.cast(j, ti.f32))) - 0.5
            c += GRAIN_STRENGTH * gn
        c = ti.math.clamp(c, 0.0, 1.0)
        fields.disp[i, j] = c * vec3(OUTPUT_TINT[0], OUTPUT_TINT[1], OUTPUT_TINT[2])


@ti.kernel
def k_expose(e: ti.f32):
    """×e 后高光处理：HIGHLIGHT_SOFT=0 硬 saturate（原版死白）；=1 软滚降 1-e^-x（高光保留层次）。"""
    for i, j in fields.pixels:
        x = ti.max(fields.pixels[i, j] * e, vec3(0.0, 0.0, 0.0))
        hard = ti.min(x, vec3(1.0, 1.0, 1.0))
        soft = vec3(1.0, 1.0, 1.0) - ti.math.exp(-x)
        fields.pixels[i, j] = lerp(hard, soft, HIGHLIGHT_SOFT)


@ti.kernel
def k_anamorphic(src: ti.template(), strength: ti.f32,
                 tr: ti.f32, tg: ti.f32, tb: ti.f32):
    """横向 1D 宽模糊镜头拉丝，加权累加到 bloom_acc。"""
    nx = fields.bloom_acc.shape[0]
    ny = fields.bloom_acc.shape[1]
    sx = src.shape[0]
    sy = src.shape[1]
    for i, j in fields.bloom_acc:
        u = (i + 0.5) / nx * sx
        v = (j + 0.5) / ny * sy
        acc = vec3(0.0, 0.0, 0.0)
        wsum = 0.0
        ANAM_TAPS = 12          # 半径 tap 数（±12=25 tap）；在 src 分辨率上的横向扩展
        ANAM_SIGMA2 = 40.0      # 高斯 σ²（tap 空间），控制软硬
        ANAM_STEP = 1.5         # 每 tap 横向步长（源纹理坐标）
        for k in range(-ANAM_TAPS, ANAM_TAPS + 1):
            w = ti.exp(-(k * k) / ANAM_SIGMA2)
            acc += bsample(src, u + k * ANAM_STEP, v) * w
            wsum += w
        acc = acc / wsum
        fields.bloom_acc[i, j] += acc * strength * vec3(tr, tg, tb)


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
    if ANAMORPHIC_ON == 1:
        assert BLOOM_LEVELS >= 3, "ANAMORPHIC 需 BLOOM_LEVELS>=3（拉丝取 bloom_lv[2] 宽层）"
        # bloom_lv[2]=RES/8 宽层：拉丝更宽更电影化（lv[0] 会偏锐利、不够"镜头感"）
        k_anamorphic(fields.bloom_lv[2], ANAMORPHIC_STRENGTH,
                     ANAMORPHIC_TINT[0], ANAMORPHIC_TINT[1], ANAMORPHIC_TINT[2])
    k_composite(BLOOM_STRENGTH, EXPOSURE_POST)   # 加 bloom → ×POST → tonemap（= 原版 Image）
