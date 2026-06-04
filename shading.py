"""
噪声/尘埃色纹理通过 `fields.noise_tex` / `fields.organic_tex` 访问（build 后才有值）。
"""
import math
import taichi as ti

from config import (vec2, vec3, vec4, NOISE_RES, ORGANIC_RES, SPIN,
                    DISK_RIN, DISK_ROUT, GLOW_FALLOFF, INNER_GLOW, DISK_TINT,
                    DUST_BRIGHT, DUST_DENSITY, TIME_SPEED, HAZE_STRENGTH, ORGANIC_ON,
                    DISK_WARP, DISK_SHEAR, DISK_FILAMENT, DISK_OCTAVE_BOOST_EFF,
                    INNER_HALO, INNER_VEIL,
                    STARMAP_ON, STARMAP_W, STARMAP_H, STARMAP_GAIN, STARMAP_ROT,
                    STARMAP_FLIP, STARMAP_PROC_MIX)
from common import lerp
import fields


@ti.func
def smoothstep2(e0: ti.f32, e1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.min(ti.max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def hash21(p: vec2) -> ti.f32:
    return ti.math.fract(ti.sin(p.dot(vec2(127.1, 311.7))) * 43758.5453)


@ti.func
def _proc_sky(theta: ti.f32, phi: ti.f32) -> vec3:
    """程序化星空：暗背景 + 网格哈希撒点星 + 暗弱银河带。"""
    uv = vec2(phi / (2.0 * math.pi), theta / math.pi)
    col = vec3(0.006, 0.008, 0.018)
    band = ti.exp(-((theta - math.pi / 2) ** 2) / 0.05)
    col += vec3(0.03, 0.026, 0.04) * band
    grid = 600.0
    g = uv * vec2(grid, grid * 0.5)
    cell = ti.floor(g)
    f = g - cell
    sp = vec2(hash21(cell), hash21(cell + vec2(5.2, 1.3)))
    d = (f - sp).norm()
    bright = hash21(cell + vec2(9.1, 2.7))
    star = ti.exp(-d * d / 0.0006) * ti.max(0.0, bright - 0.55) / 0.45
    tint = vec3(0.8 + 0.2 * hash21(cell + vec2(3.0, 7.0)), 0.85,
                0.9 + 0.1 * hash21(cell + vec2(7.0, 3.0)))
    col += star * tint * 3.0
    return col


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


@ti.func
def pcurve(x: ti.f32, a: ti.f32, b: ti.f32) -> ti.f32:
    k = ti.pow(a + b, a + b) / (ti.pow(a, a) * ti.pow(b, b))
    xx = ti.min(ti.max(x, 0.0), 1.0)
    return k * ti.pow(xx, a) * ti.pow(1.0 - xx, b)


@ti.func
def sample_noise_rg(uv: vec2) -> vec2:
    p = uv - ti.floor(uv / NOISE_RES) * NOISE_RES
    i0 = int(ti.floor(p.x))
    j0 = int(ti.floor(p.y))
    fx = p.x - i0
    fy = p.y - j0
    i0 %= NOISE_RES
    j0 %= NOISE_RES
    i1 = (i0 + 1) % NOISE_RES
    j1 = (j0 + 1) % NOISE_RES
    a = fields.noise_tex[i0, j0]
    b = fields.noise_tex[i1, j0]
    c = fields.noise_tex[i0, j1]
    d = fields.noise_tex[i1, j1]
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy)


@ti.func
def noise3(x: vec3) -> ti.f32:
    """iq 3D 值噪声：RGBA 图 R/G 两通道沿 z 插值。"""
    p = ti.floor(x)
    f = ti.math.fract(x)
    f = f * f * (3.0 - 2.0 * f)
    uvx = p[0] + 37.0 * p[2] + f[0]
    uvy = p[1] + 17.0 * p[2] + f[1]
    s = sample_noise_rg(vec2(uvx, uvy))
    return -1.0 + 2.0 * lerp(s[1], s[0], f[2])


@ti.func
def sample_organic(uv: vec2) -> vec3:
    p = uv - ti.floor(uv)
    g = p * ORGANIC_RES
    i0 = int(ti.floor(g.x))
    j0 = int(ti.floor(g.y))
    fx = g.x - i0
    fy = g.y - j0
    i0 %= ORGANIC_RES
    j0 %= ORGANIC_RES
    i1 = (i0 + 1) % ORGANIC_RES
    j1 = (j0 + 1) % ORGANIC_RES
    a = fields.organic_tex[i0, j0]
    b = fields.organic_tex[i1, j0]
    c = fields.organic_tex[i0, j1]
    d = fields.organic_tex[i1, j1]
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy)


@ti.func
def gas_disc_sample(pos: vec3, ds: ti.f32, tparam: ti.f32) -> vec4:
    """Buffer A GasDisc 体积采样，返回 vec4(预乘发射色 rgb, 覆盖率 cov)。pos 为笛卡尔，盘在 z=0。"""
    SCL = 5.3 / (DISK_ROUT - DISK_RIN)
    x, y, z = pos[0], pos[1], pos[2]
    rho_cyl = ti.sqrt(x * x + y * y)
    dc = (rho_cyl - DISK_RIN) * SCL + 0.55
    dd = z * SCL
    emis = vec3(0.0, 0.0, 0.0)
    cov = 0.0
    rg = 1.0 - ti.min(ti.max((dc - 0.55) / 5.3 * 0.5, 0.0), 1.0)
    if rg > 0.001:
        cov0 = pcurve(rg, 4.0, 0.9)
        dthick = 0.1 * rg
        cov0 *= ti.max(1.0 - ti.abs(dd) / ti.max(dthick, 1e-4), 0.0)
        dust_glow = 1.0 / (ti.pow(1.0 - rg, 2.0) * GLOW_FALLOFF + 0.002)
        dust = vec3(1.0, 1.0, 1.0) * dust_glow * 8.2
        cov0 = ti.min(ti.max(cov0 * 0.7, 0.0), 1.0)
        fade = ti.pow(ti.abs(dc - 0.55) + 0.4, 4.0) * 0.04
        bloom_f = 1.0 / (dd * dd * 40.0 + fade + 0.00002)
        bcol = vec3(1.0, 1.0, 1.0) * ti.pow(bloom_f, 1.5)
        bcol *= lerp(vec3(1.7, 1.1, 1.0), vec3(0.5, 0.6, 1.0), rg * rg)
        bcol *= lerp(vec3(1.7, 0.5, 0.1), vec3(1.0, 1.0, 1.0), ti.sqrt(rg))
        dust = lerp(dust, bcol * 150.0 * INNER_GLOW, ti.min(ti.max(1.0 - cov0, 0.0), 1.0))
        # 内缘分层：中晕（rg≈0.57 峰值）+ 宽幅泛光（rg≈0.33 峰值，更靠内缘）。默认强度 0，恒等回归
        # 系数 30/8 为相对基础内缘辉光 INNER_GLOW(bcol×150) 的经验调谐幅度，随旋钮线性缩放
        dust += bcol * 30.0 * INNER_HALO * pcurve(rg, 2.0, 1.5)
        dust += bcol * 8.0 * INNER_VEIL * pcurve(rg, 1.0, 2.0)
        cov0 = ti.min(ti.max(cov0 + bloom_f * bloom_f * 0.1, 0.0), 1.0)
        if cov0 >= 0.01:
            ang = ti.atan2(y, x)          # 切口在 -x（黑洞背后），避开正前方那条接缝
            rcx = (dc * 1.5 + 0.55) * 0.95
            # 方位各向异性：DISK_FILAMENT>1 时沿切向拉长结构（除以比值=降方位频率）。=1 时恒等
            rcy = (ang * 1.5) * 0.95 / ti.max(DISK_FILAMENT, 1e-3)
            rcz = (dd * 1.5) * 0.95
            # 较差自转：内快外慢的相位增量。DISK_SHEAR=0 时退化为原版均匀 sft
            # 指数 1.5 ≈ 开普勒 Ω∝r^(-3/2)；shear 在 rho=DISK_RIN 处为 0（内缘不加速），外缘趋近 -1
            shear = (DISK_RIN / ti.max(rho_cyl, DISK_RIN)) ** 1.5 - 1.0
            sft = tparam * TIME_SPEED * (1.0 + DISK_SHEAR * shear)
            # 域扭曲：低频噪声向量扰动采样坐标。DISK_WARP=0 时 w=0，恒等
            # *0.5=低频（扭曲旋涡比湍流大 2 倍）；两组奇数偏移使 xyz 三通道去相关
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
            dust *= n1 * 0.998 + 0.002
            cov0 *= n2
            if ti.static(ORGANIC_ON == 1):
                # 注：rcy 在 DISK_FILAMENT!=1 时已被各向异性缩放，organic 的 uvo 同步拉伸（同向一致，预期行为）
                uvo = vec2((rcy + tparam * TIME_SPEED * 0.5) * 0.15, rcx * 0.27)
                org = sample_organic(uvo)
                dust *= ti.pow(org, vec3(2.0, 2.0, 2.0)) * 4.0 * DUST_BRIGHT
            else:
                dust *= lerp(vec3(1.6, 0.7, 0.3), vec3(1.0, 1.0, 1.0), ti.sqrt(rg)) * DUST_BRIGHT
            cov0 = ti.min(ti.max(cov0 * ds * SCL * DUST_DENSITY, 0.0), 1.0)
            dust = ti.max(dust, vec3(0.0, 0.0, 0.0))
            cov0 *= pcurve(rg, 4.0, 0.9)
            emis = dust * vec3(DISK_TINT[0], DISK_TINT[1], DISK_TINT[2])
            cov = cov0
    return vec4(emis[0], emis[1], emis[2], cov)


@ti.func
def haze_sample(pos: vec3, alpha: ti.f32, ds: ti.f32) -> vec3:
    SCL = 5.3 / (DISK_ROUT - DISK_RIN)
    x, y, z = pos[0], pos[1], pos[2]
    rho_cyl = ti.sqrt(x * x + y * y)
    dc = (rho_cyl - DISK_RIN) * SCL + 0.55
    dd = z * SCL
    qx = dc - 1.0
    td = ti.abs(ti.sqrt(qx * qx + dd * dd) - 0.01)
    bloom_disc = 1.0 / (td * td + 0.001)
    gate = 0.0
    if dc > 0.5:
        gate = 1.0
    return vec3(1.0, 1.0, 1.0) * bloom_disc * gate * (2.9 / 15.0) * ds * SCL \
        * (1.0 - alpha) * HAZE_STRENGTH


@ti.func
def disc_blueshift(r: ti.f32, b: ti.f32) -> ti.f32:
    """赤道顺行圆轨道发射体的频移因子 g（= 原版）。"""
    a = SPIN
    sq = ti.sqrt(r)
    Om = 1.0 / (r * sq + a)
    denom = ti.max(r * r - 3.0 * r + 2.0 * a * sq, 1e-4)
    ut = (r + a / sq) / ti.sqrt(denom)
    g = 1.0 / (ut * (1.0 - Om * b))
    return ti.min(ti.max(g, 0.1), 4.0)
