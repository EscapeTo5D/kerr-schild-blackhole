"""
全局缓冲 field 持有 + 分配（build）+ 纹理加载。

其它模块通过 `import fields` 后用 `fields.pixels` 等访问；不要用 `from fields import pixels`
（build() 重新赋值后，import 进来的名字不会更新）。Taichi 在 kernel 编译时（build 之后）
解析 `fields.xxx`，拿到真正的 field。
"""
import numpy as np
import taichi as ti

from config import (vec3, RES_X, RES_Y, NOISE_RES, ORGANIC_RES, BLOOM_LEVELS,
                    STARMAP_W, STARMAP_H, STARMAP_ON)

pixels = None        # HDR 渲染缓冲（超采样平均后）
samp_acc = None      # 超采样累加缓冲
disp = None          # 最终显示缓冲（bloom + tonemap 后）
bloom_acc = None     # 全分辨率 bloom 累加器
bloom_lv = []        # 各级降采样后的 bloom
noise_tex = None     # 256² 噪声纹理（= 原版 iChannel0）
organic_tex = None   # 1024² 尘埃色纹理（= 原版 iChannel1 = Organic 1）
starmap_tex = None   # 等距柱状星图纹理（= C 任务底图）


def build():
    """按 config 的 RES/BLOOM_LEVELS 分配所有缓冲（首次渲染前调用一次）。"""
    global pixels, samp_acc, disp, bloom_acc, bloom_lv, noise_tex, organic_tex, starmap_tex
    noise_tex = ti.Vector.field(2, dtype=ti.f32, shape=(NOISE_RES, NOISE_RES))
    organic_tex = ti.Vector.field(3, dtype=ti.f32, shape=(ORGANIC_RES, ORGANIC_RES))
    if STARMAP_ON == 1:
        starmap_tex = ti.Vector.field(3, dtype=ti.f32, shape=(STARMAP_W, STARMAP_H))
    else:
        starmap_tex = ti.Vector.field(3, dtype=ti.f32, shape=(1, 1))
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
    samp_acc = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
    disp = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
    bloom_acc = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
    bloom_lv = []
    for i in range(BLOOM_LEVELS):
        w = max(RES_X >> (i + 1), 1)
        h = max(RES_Y >> (i + 1), 1)
        bloom_lv.append(ti.Vector.field(3, dtype=ti.f32, shape=(w, h)))


def load_noise_tex(path=None):
    """填充 256² 双通道（R,G）噪声纹理（iq 的 noise(vec3) 用 R/G 两通道造 3D 噪声）。"""
    def resample(ch):
        h, w = ch.shape
        yi = (np.arange(NOISE_RES) * h // NOISE_RES) % h
        xi = (np.arange(NOISE_RES) * w // NOISE_RES) % w
        return np.ascontiguousarray(ch[np.ix_(yi, xi)].T, dtype=np.float32)

    if path:
        img = ti.tools.imread(path).astype(np.float32) / 255.0
        if img.ndim == 3:
            r = img[..., 0]
            g = img[..., 1] if img.shape[2] >= 2 else img[..., 0]
        else:
            r = g = img
        grid = np.stack([resample(r), resample(g)], axis=-1)
    else:
        rng = np.random.default_rng(12345)
        r = rng.random((NOISE_RES, NOISE_RES), dtype=np.float32)
        g = rng.random((NOISE_RES, NOISE_RES), dtype=np.float32)
        grid = np.stack([r, g], axis=-1)
    noise_tex.from_numpy(np.ascontiguousarray(grid, dtype=np.float32))


def load_organic_tex(path=None):
    """填充 1024² 尘埃色纹理（= 原版 iChannel1 = Organic 1）。"""
    if path:
        img = ti.tools.imread(path).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img[..., :3]
        h, w = img.shape[0], img.shape[1]
        yi = (np.arange(ORGANIC_RES) * h // ORGANIC_RES) % h
        xi = (np.arange(ORGANIC_RES) * w // ORGANIC_RES) % w
        grid = img[np.ix_(yi, xi)]
        grid = np.ascontiguousarray(np.transpose(grid, (1, 0, 2)), dtype=np.float32)
    else:
        grid = np.full((ORGANIC_RES, ORGANIC_RES, 3), 0.5, dtype=np.float32)
    organic_tex.from_numpy(grid)


def load_starmap_tex(path=None):
    """填充等距柱状星图，field 形状 (STARMAP_W 方位, STARMAP_H 极向)。
    注意：ti.tools.imread 返回 (W,H,C)，方位轴在 dim0、极向轴在 dim1，
    与 field 轴序一致，故直接按轴索引、无需 transpose（区别于 organic 的正方形 loader）。"""
    if path:
        img = ti.tools.imread(path).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img[..., :3]
        w_img, h_img = img.shape[0], img.shape[1]   # W=方位, H=极向
        xi = (np.arange(STARMAP_W) * w_img // STARMAP_W) % w_img
        yi = (np.arange(STARMAP_H) * h_img // STARMAP_H) % h_img
        grid = np.ascontiguousarray(img[np.ix_(xi, yi)], dtype=np.float32)  # (STARMAP_W, STARMAP_H, 3)
    else:
        grid = np.zeros((STARMAP_W, STARMAP_H, 3), dtype=np.float32)
    starmap_tex.from_numpy(grid)
