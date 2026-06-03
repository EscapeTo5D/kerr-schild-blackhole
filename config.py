"""
interstellar_ks · 运行配置 + Taichi 初始化 + 公共类型别名。

在 IDE 里改这里的参数，然后直接 Run `interstellar_ks/main.py` 即可（无需命令行）。
所有标量常量被各模块以 `from config import *` 引入；进程重启即重新绑定，改完重跑就生效。
"""
import os

import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True, device_memory_fraction=0.9)

# 项目根（= 本仓库根，即这些 .py 与 assets/ 所在目录）。相对路径都锚到这里，不随 IDE 工作目录变化。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _rel(path):
    """相对路径 → 锚定到项目根的绝对路径；绝对路径/None 原样返回。"""
    return os.path.join(PROJECT_ROOT, path) if path and not os.path.isabs(path) else path

# 公共向量/矩阵类型别名（各模块共用）
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.types.vector(4, ti.f32)
mat4 = ti.types.matrix(4, 4, ti.f32)

# ============================ 运行配置（改这里，点 Run）============================
RES_X, RES_Y = 3840, 2160            # 输出分辨率（4K 动画；出 8K 大图改回 8192,4320）
OUT_PATH = _rel("interstellar_ks_4k.png")  # 输出文件名（写相对路径即锚定项目根）
INTERACTIVE = False                 # True=实时窗口；False=出静帧
NOISE_TEX_PATH = _rel("assets/rgba_noise_medium.png")   # iChannel0 噪声图，None=白噪声
ORGANIC_TEX_PATH = _rel("assets/organic1.jpg")          # iChannel1 尘埃色，None=暖色近似（同时关 organic 分支）

# ------------------------------------------------------------------ 物理 / 相机
SPIN = 0.6                          # 黑洞自旋 a/M
CAM_DIST = 30.0                     # 相机半径
CAMERA_THETA = 0.08                # 相机相对赤道面仰角（弧度）
CAMERA_PHI = 0.0
FOV_FACTOR = 1.2                   # 视野（越大越窄）
TIME_OFFSET = 0.2                  # 噪声初始相位

# ------------------------------------------------------------------ 吸积盘（Buffer A 体积盘）
DISK_RIN = 3.83                    # 内缘
DISK_ROUT = 15.0                   # 外缘
DISK_ON = True
DISK_HALF_Z = 0.30                 # 盘附近限步/密度存在的竖直半厚（M）
DISK_STEP = 0.05                   # 进入盘包围层后的最大步长（M）
DUST_DENSITY = 80.0                # 单位路径长 coverage 增益（= 原版 1200/ITER 折算）
DUST_BRIGHT = 1.0                  # 盘整体亮度额外增益（1.0=与原版一致；原版的 ×4 已在 organic 分支内）
GLOW_FALLOFF = 10.0                # 盘发射径向衰减锐度（dust_glow 分母）：越小外缘越亮、两翼延伸越远；290=原版
DISK_TINT = (1.0, 0.78, 0.55)      # 盘发射色染色（红移/色温）：G、B 越低越红；(1,1,1)=不染色保持原版白
INNER_GLOW = 1.0                   # 内缘强自发光（bcol×150）增益（1.0=原版 lstSRS）
HAZE_STRENGTH = 1.0                # 内缘辉光（原版 Haze；1.0=原版 lstSRS；0=关）
DISK_DOPPLER = 0                   # 原版 lstSRS 假弯曲无多普勒，故 0=与原版一致；1=加真实多普勒
DOPPLER_POWER = 2.0                # 多普勒指数：g²（柔和）/ g³（原版更强）
DOPPLER_TILT = 1.0                 # 多普勒强度随相机仰角的 cos²θ 调制（借 canvas 版思路）：1=edge-on 满/face-on 无；0=不随仰角变（纯 POWER）。仅 DISK_DOPPLER=1 时生效
TIME_SPEED = 0.06                  # 噪声沿方位随 t 旋进

NOISE_RES = 256                    # iChannel0 噪声纹理 256²（双通道 R/G）
ORGANIC_RES = 1024                 # iChannel1 尘埃色原生 1024²（Organic 1 原尺寸，不降采样）
# organic 分支开关：由 ORGANIC_TEX_PATH 自动派生（被 ti.static 用作编译期常量）。
# 想关 organic 改用暖色近似 → 把 ORGANIC_TEX_PATH 设为 None。
ORGANIC_ON = 1 if ORGANIC_TEX_PATH else 0

# ------------------------------------------------------------------ 积分器（KS RK45）
MAX_STEPS = 20000
INT_STEP = 0.5                     # RK45 初始步长
RK_RTOL = 1.0e-6
RK_ATOL = 1.0e-8
RK_H_MIN = 1.0e-4
RK_H_MAX = 4.0
RK_MAX_REJECT = 8

# ------------------------------------------------------------------ 后处理（Bloom + filmic）
BLOOM_LEVELS = 8                   # = 原版 lstSRS 的 8 个 octave（到 1/256，含最宽柔光晕）
BLOOM_STRENGTH = 0.07              # bloom 叠加强度（= 原版 GetBloom*0.08 量级）
# 两段曝光（复刻原版顺序：×PRE 高光处理 → 加 bloom → ×POST → tonemap）
EXPOSURE_PRE = 0.01                # 钳位前缩放（净曝光≈PRE×POST=0.05）
EXPOSURE_POST = 5.0                # 钳位后增益
CONTRAST = 1.0                     # tonemap S 曲线（smoothstep）强度：1=原版增对比；<1 压缩动态范围（高光降、暗部抬）；0=不增对比
HIGHLIGHT_SOFT = 1.0               # 高光处理：0=硬 saturate（原版，高光死白）；1=软滚降 1-e^-x（高光不死白、保留层次）
OUTPUT_TINT = (1.0, 0.82, 0.62)    # 最终输出整体色调（LDR 域，连白区也染）：R 保持 1、压 G/B 即偏橙；(1,1,1)=中性
SAMPLES = 2                        # TAA 超采样数；1=关（动画临时降配；出大图改回 8）

R_SKY = 60.0                       # 逃逸半径（KS 下偏折发生在近黑洞处，60 足够）

# ------------------------------------------------------------------ 动画（序列帧 → ffmpeg 合成）
ANIMATE = True                     # True=出序列帧（覆盖单帧出图）；INTERACTIVE 优先级更高
ANIM_FRAMES = 60                    # 总帧数（时长 = ANIM_FRAMES / ANIM_FPS = 60/12 = 5s）
ANIM_FPS = 12                       # 合成帧率（60帧@12fps=5s；想更顺改 120帧@24fps，同样 5s）
ANIM_CAM_TURNS = 0.0               # 相机绕黑洞转几圈（0=相机不动，只盘流动）
ANIM_DISK_TURN = 30.0              # 盘流动总量 = t 的总推进；噪声相位总移 = 本值×TIME_SPEED(0.06)=1.8。流动速度 = 1.8/5s = 0.36/s（慢）
ANIM_OUT_DIR = _rel("frames_ks")   # 序列帧输出目录（锚定项目根）
ANIM_AUTO_MERGE = True             # 跑完自动调 ffmpeg 合成 mp4（需系统装了 ffmpeg，否则回退为打印命令）
ANIM_VIDEO = _rel("interstellar_ks.mp4")   # 合成视频输出路径
# 注意：动画务必降分辨率/采样，否则 8K×120 帧会跑到天荒地老。
# 建议出动画时临时设 RES_X,RES_Y = 1920,1080 且 SAMPLES = 2~4。
