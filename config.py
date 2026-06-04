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
RES_X, RES_Y = 3840, 2160            # 出图/视频分辨率（INTERACTIVE=False 时引擎用这个；4K，出 8K 大图改 8192,4320）
OUT_PATH = _rel("interstellar_ks_4k.png")  # 输出文件名（写相对路径即锚定项目根）
INTERACTIVE = False               # True=实时 GUI 窗口（调参用）；False=出静帧/视频
NOISE_TEX_PATH = _rel("assets/rgba_noise_medium.png")   # iChannel0 噪声图，None=白噪声
ORGANIC_TEX_PATH = _rel("assets/organic1.jpg")          # iChannel1 尘埃色，None=暖色近似（同时关 organic 分支）

# ============================ 电影化预设（推荐值，默认不启用）============================
# 下面所有增强旋钮默认都取「原版/关闭」值，开箱观感与原版逐像素一致（便于回归对比）。
# 想要一轮完整的「电影化」观感，把下面这组推荐值填到各对应旋钮（保留 TONEMAP=0 的深黑招牌色调）：
#   丝状湍流：   DISK_WARP=0.6  DISK_SHEAR=0.8  DISK_FILAMENT=1.6  DISK_OCTAVE_BOOST=1
#   光子环/内缘： PHOTONRING_ON=1  PHOTONRING_STRENGTH=2.0  INNER_HALO=1.0  INNER_VEIL=0.6
#   白热盘体：   INNER_GLOW=3.0  WHITE_BASE_POW=2.0  WHITE_SIZE=0.001  （饱满白热 + 紧凑收束；原版=1.0/4.0/0.0）
#   星图背景：   STARMAP_GAIN=1.8  STARMAP_PROC_MIX=0.25  STARMAP_FLIP=1  （FLIP=1 抵消 imread 翻转，银河北极朝上）
#   后期：       TONEMAP=0  ANAMORPHIC_ON=1  ANAMORPHIC_STRENGTH=0.8  VIGNETTE_STRENGTH=0.3
# 逐项可调；想更狠把 DISK_*/PHOTONRING_STRENGTH/STARMAP_GAIN 调大。GUI 实时档会自动跳过额外 octave。

# -------- GUI 交互调参专用（仅 INTERACTIVE=True 生效；改这里绝不影响上面的出图/视频参数）--------
# GUI 用这套低配跑得动、跟手；调完把 INTERACTIVE 设回 False，出图/视频自动切回上面的高配。
GUI_RES_X, GUI_RES_Y = 1280, 720     # 交互窗口分辨率（小=流畅）
GUI_SAMPLES = 1                      # 交互每帧 TAA 采样（1=最快最跟手；2~4=更干净但更卡）
GUI_TIME_STEP = 0.3                  # 交互每帧盘流动推进量（噪声相位/帧 = 本值×TIME_SPEED）；调大流得快、调小更平滑
GUI_ZOOM_STEP = 1.1                  # Ctrl+滚轮每格的视角缩放倍率（>1）；1.1=每格放大/缩小 10%，越大越灵敏

# ------------------------------------------------------------------ 物理 / 相机
SPIN = 0.6                          # 黑洞自旋 a/M
CAM_DIST = 24.1                     # 相机半径（拉近出前景沉浸感；30=原版远景）
CAMERA_THETA = 0.081               # 相机相对赤道面仰角（弧度）
CAMERA_PHI = 0.47                  # 方位角（盘轴对称，改此不改构图，只转噪声相位/错开正面直纹接缝）
FOV_FACTOR = 1.50                  # 视野（越大越窄）
TIME_OFFSET = 0.2                  # 噪声初始相位
# 电影构图（《星际》斜构图：黑洞右上、盘对角线）。全取 0 + CAM_DIST=30 = 原版居中正水平 edge-on：
CAMERA_ROLL = -0.260               # 画面滚转（绕视线轴，弧度）：盘斜成对角线
CAMERA_PAN_X = 0.160               # 主点水平偏移（屏幕单位 ux∈[-1,1]）：黑洞横向位置；>0 黑洞右移
CAMERA_PAN_Y = 0.160               # 主点竖直偏移：黑洞纵向位置；>0 黑洞上移

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
INNER_GLOW = 3.0                   # 内缘强自发光（bcol×150）增益（1.0=原版 lstSRS）
# 内缘白热亮度函数（解耦「亮度」与「尺寸」，见 shading.gas_disc_sample）：
WHITE_BASE_POW = 2.0               # 白热径向衰减幂：4.0=原版细环（回归安全）；越小越饱满白热（2.0=whitebright 式）
WHITE_SIZE = 0.001                 # 白热外缘截断墙系数：0.0=原版无截断（回归安全）；>0 截住白光半径（越大越紧），与 BASE_POW 解耦控亮/控大
# ---- B 光子环 / 内缘分层（默认关闭/原版）----
PHOTONRING_ON = 1            # 0=关（回归安全）；1=叠加刀锋光子环
PHOTONRING_STRENGTH = 1.0    # 光子环亮度
PHOTONRING_WIDTH = 0.15      # 光子环径向宽度（高斯 sigma，单位 M）
PHOTONRING_TINT = (1.0, 0.9, 0.75)   # 光子环色染
PHOTONRING_RADIUS = -1.0     # 光子球半径；<=0 则按 SPIN 自动算（顺行赤道光子轨道）
INNER_HALO = 1.0             # 内缘中晕强度（0=原版无此层）
INNER_VEIL = 0.6             # 内缘宽幅泛光强度（0=原版无此层）
HAZE_STRENGTH = 1.0                # 内缘辉光（原版 Haze；1.0=原版 lstSRS；0=关）
DISK_DOPPLER = 0                   # 原版 lstSRS 假弯曲无多普勒，故 0=与原版一致；1=加真实多普勒
DOPPLER_POWER = 2.0                # 多普勒指数：g²（柔和）/ g³（原版更强）
DOPPLER_TILT = 1.0                 # 多普勒强度随相机仰角的 cos²θ 调制（借 canvas 版思路）：1=edge-on 满/face-on 无；0=不随仰角变（纯 POWER）。仅 DISK_DOPPLER=1 时生效
TIME_SPEED = 0.06                  # 噪声沿方位随 t 旋进

# ---- A 丝状湍流（默认全取原版值）----
DISK_WARP = 0.6          # 域扭曲强度：低频噪声扰动采样坐标，造流体丝状旋涡。0=原版块状
DISK_SHEAR = 0.8         # 较差自转剪切：噪声方位相位随 r 变化（内快外慢），丝拖成螺旋。0=均匀=原版
DISK_FILAMENT = 1.6      # 方位各向异性拉伸比（>0，建议 0.5~4）：沿切向拉长结构顺流线。1=各向同性=原版
DISK_OCTAVE_BOOST = 1    # 离线额外高频 octave 数（0/1/2）。0=原版层数；GUI 档强制 0

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
CONTRAST = 1.0                     # tonemap S 曲线（smoothstep）强度：1=原版增对比；<1 压缩动态范围（高光降、暗部抬）；0=不增对比（仅 TONEMAP=0 生效；ACES/AgX 自带 S 形对比）
HIGHLIGHT_SOFT = 1.0               # 高光处理：0=硬 saturate（原版，高光死白）；1=软滚降 1-e^-x（高光不死白、保留层次）
OUTPUT_TINT = (1.0, 0.82, 0.62)    # 最终输出整体色调（LDR 域，连白区也染）：R 保持 1、压 G/B 即偏橙；(1,1,1)=中性
SAMPLES = 2                        # TAA 超采样数；1=关（动画临时降配；出大图改回 8）

# ---- D 后期电影化（默认全取原版/关闭）----
TONEMAP = 0                  # 0=现有曲线 / 1=ACES / 2=AgX 近似
ANAMORPHIC_ON = 0            # 1=横向镜头拉丝
ANAMORPHIC_STRENGTH = 0.5    # 拉丝强度
ANAMORPHIC_TINT = (0.4, 0.6, 1.0)  # 拉丝色染（经典冷蓝）
VIGNETTE_STRENGTH = 0.0      # 暗角强度（0=关）
GRAIN_STRENGTH = 0.0         # 胶片颗粒幅度（0=关）
CHROMA_AB_STRENGTH = 0.0     # 边缘色散强度（0=关）

R_SKY = 60.0                       # 逃逸半径（KS 下偏折发生在近黑洞处，60 足够）

# ---- C 图像底图星空 ----
STARMAP_ENABLE = True        # 天空盒总开关：False=关图像底图，回退纯程序化星空（_proc_sky）。关时星图纹理不加载、不占显存
STARMAP_PATH = _rel("assets/milkyway_eso0932a.jpg")  # None=回退纯程序化星空（回归安全）
STARMAP_RES = (4096, 2048)   # 星图采样分辨率（W,H，2:1）。显存=W*H*3*4 字节≈100MB
STARMAP_GAIN = 1.0           # 底图亮度增益
STARMAP_ROT = 0.0            # 方位旋转（绕极轴，单位"圈"，对准银河）
STARMAP_FLIP = 0             # 1=上下翻转 v（抵消 imread 的 H 轴翻转；标准「北极在上」图须设 1）
STARMAP_PROC_MIX = 0.3       # 叠加程序化星空（含星点/暗背景/程序带）的权重；0=纯图。注：叠的是整张 _proc_sky 非仅亮星

# ------------------------------------------------------------------ 动画（序列帧 → ffmpeg 合成）
ANIMATE = False                     # True=出序列帧（覆盖单帧出图）；INTERACTIVE 优先级更高
ANIM_FRAMES = 60                    # 总帧数（时长 = ANIM_FRAMES / ANIM_FPS = 60/12 = 5s）
ANIM_FPS = 12                       # 合成帧率（60帧@12fps=5s；想更顺改 120帧@24fps，同样 5s）
ANIM_CAM_TURNS = 0.0               # 相机绕黑洞转几圈（0=相机不动，只盘流动）
ANIM_DISK_TURN = 30.0              # 盘流动总量 = t 的总推进；噪声相位总移 = 本值×TIME_SPEED(0.06)=1.8。流动速度 = 1.8/5s = 0.36/s（慢）
ANIM_OUT_DIR = _rel("frames_ks")   # 序列帧输出目录（锚定项目根）
ANIM_AUTO_MERGE = True             # 跑完自动调 ffmpeg 合成 mp4（需系统装了 ffmpeg，否则回退为打印命令）
ANIM_VIDEO = _rel("interstellar_ks.mp4")   # 合成视频输出路径
# 注意：动画务必降分辨率/采样，否则 8K×120 帧会跑到天荒地老。
# 建议出动画时临时设 RES_X,RES_Y = 1920,1080 且 SAMPLES = 2~4。

# ==================== 模式派生（计算项，不要在这里改参数）====================
# GUI 模式下，用上面的 GUI 专用低配覆盖引擎实际使用的分辨率/采样；
# 出图/视频模式（INTERACTIVE=False）保持原高配不变 —— 因此调 GUI_* 永远不会动到视频参数。
if INTERACTIVE:
    RES_X, RES_Y = GUI_RES_X, GUI_RES_Y
    SAMPLES = GUI_SAMPLES

# A：离线才加额外 octave；GUI 实时档强制 0（编译期常量，被 ti.static 用）
DISK_OCTAVE_BOOST_EFF = 0 if INTERACTIVE else DISK_OCTAVE_BOOST

# C：星图是否启用（编译期常量，被 ti.static 用）+ 尺寸常量。开关与路径同时满足才启用
STARMAP_ON = 1 if (STARMAP_ENABLE and STARMAP_PATH) else 0
STARMAP_W, STARMAP_H = STARMAP_RES
