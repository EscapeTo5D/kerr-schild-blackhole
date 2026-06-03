# 电影化增强 · 设计 spec

日期：2026-06-04
项目：interstellar_ks（Kerr-Schild 黑洞 Taichi/GPU 渲染器）
目标方向：**更强的电影观感**（非物理建模）

## 背景与约束

当前吸积盘是移植自 Shadertoy（lstSRS "interstellar"）的程序化体积盘：白热尘埃（径向辉光衰减）+ 内缘强自发光 + Haze 辉光 + 程序化噪声湍流 + 可选 organic 染色；背景为纯程序化星空。

本次为一轮**纯视觉电影化增强**，遵循以下约束：

- **不动物理与积分器**：不引入 Novikov-Thorne 温度剖面、不做光谱多普勒、不改 RK45 / 测地线引擎。性能与数值精度是另一个独立 spec 的范畴。
- **离线渲染可任意慢**：4K/8K 静帧与视频是离线渲染，单帧时间翻倍可接受，画质优先。
- **GUI 实时档自动降级**：新效果在 GUI 模式下通过 `ti.static` 质量开关或 `GUI_*` 覆盖自动降级/关闭，保持实时跟手。
- **沿用现有架构**：所有新行为通过 `config.py` 旋钮暴露；新缓冲/贴图进 `fields.py`；逻辑进 `shading.py` / `postprocess.py`。模块边界不变。

现有模块边界（保持不变）：
`config.py`（常量）→ `fields.py`（缓冲/贴图）→ `geodesic.py`（积分器）→ `shading.py`（盘/天）→ `render.py`（kernel + 驱动）→ `postprocess.py`（后期）→ `main.py`（入口）。

## 四条工作线

### A. 盘体丝状湍流 —— `shading.py: gas_disc_sample`

**问题**：现 `n1/n2` 是各向同性 value-noise 连乘，结构呈块状、偏糊，缺少流体丝状/旋涡的"物质旋进"读感。

**改动**：

1. **域扭曲（domain warp）**：采高频噪声前，先用一层低频噪声向量扰动采样坐标（`p += DISK_WARP * warp_vec(p_low)`），把各向同性块状重塑为流体丝状旋涡。
2. **较差自转剪切**：当前噪声方位相位偏移 `sft = tparam * TIME_SPEED` 沿半径均匀。改为**随 r 变化的角速度**（内快外慢，近开普勒定性 `ω(r) ∝ r^(-3/2)` 量级，作为艺术近似而非严格物理），使丝状结构被剪切成拖尾螺旋。仅作用于噪声采样的方位坐标，不影响盘的几何覆盖。
3. **各向异性拉伸 + 加 octave**：沿方位（切向）方向拉伸噪声采样尺度，让结构顺流线；离线档额外叠 1~2 层高频 octave 出精细丝。

**新旋钮**（`config.py`）：

- `DISK_WARP`（域扭曲强度，0=关，复刻原版）
- `DISK_SHEAR`（较差自转剪切强度，0=均匀相位=原版）
- `DISK_FILAMENT`（方位各向异性拉伸比，1=各向同性=原版）
- `DISK_OCTAVE_BOOST`（离线额外高频 octave 数，0=原版层数）

**GUI 降级**：额外 octave 与高频细节通过 `ti.static(GUI 标志)` 在 GUI 档跳过。

**回归保证**：四个旋钮全取"原版值"时，`gas_disc_sample` 输出与当前完全一致（逐项可关）。

### B. 内缘 / 光子环 —— `render.py` + `shading.py`

**问题**：内缘自发光是一团，缺少光子环的刀锋亮环与结构化层次。

**改动**：

1. **真·光子环**：在 `render.py` 的积分循环里记录每条光线的**最近接近半径 `r_min`**（循环内维护一个标量，每步 `r_min = min(r_min, r_curr)`，近零成本）。光线终止后（命中视界或逃逸），当 `r_min` 落在光子球半径附近的窄带内时，叠加一条**极细高亮环**发射项 → 黑洞剪影外那圈标志性亮环，独立于盘的体积发射。
   - 光子球半径：赤道顺/逆行光子轨道半径随自旋 `a` 变化；取一个随 `SPIN` 计算的代表值（或简化为 Schwarzschild 的 `3M` 附近并允许旋钮微调），峰值用窄高斯 `exp(-((r_min - r_ph)/WIDTH)^2)`。
   - 该项加到 `col`，受 `(1-alpha)` 透过率调制（盘前的不被盖）。
2. **内缘多层辉光**：把现 `gas_disc_sample` 内缘自发光（`bcol×150×INNER_GLOW`）拆为三层独立强度/染色：
   - **核**（紧、热）：窄径向核心高亮。
   - **中晕**：中等半径过渡晕。
   - **宽幅泛光（veil）**：宽幅低强度铺底。
   - 三层叠加替代单一 blob，产生结构化径向衰减。

**新旋钮**（`config.py`）：

- `PHOTONRING_ON`（0/1）/ `PHOTONRING_STRENGTH` / `PHOTONRING_WIDTH` / `PHOTONRING_TINT`(rgb) / `PHOTONRING_RADIUS`（光子球半径覆盖；≤0 则按 SPIN 自动算）
- `INNER_CORE` / `INNER_HALO` / `INNER_VEIL`（三层强度；三者按比例对齐原版 `INNER_GLOW` 时复刻原观感）

**回归保证**：`PHOTONRING_ON=0` 且三层强度等效原 `INNER_GLOW` 时，输出与当前一致。

### C. 图像底图星空 —— `fields.py` + `shading.py` + `config.py` + `assets/`

**问题**：纯程序化星空简陋，缺纵深，衬不出盘。

**改动**：

1. **资源**：联网获取一张开源**等距柱状（equirectangular）银河全景**（候选：NASA Deep Star Maps 2020 / ESO Milky Way panorama，授权可商用、注明来源），降采样到 `STARMAP_RES` 后存入 `assets/`。在 `README` 或 spec 记录来源与授权。
2. **加载**：`fields.py` 新增 `starmap_tex`（`ti.Vector.field(3, ...)`）+ `load_starmap_tex(path)`，照搬 `load_organic_tex` 的 imread/降采样/转置写法。`build()` 中分配，`main.main()` 中加载。
3. **采样**：`shading.py: sky_color` 改为对 `starmap_tex` 双线性采样：
   - `u = (phi/(2π) + STARMAP_ROT) mod 1`，`v = theta/π`（带翻转旋钮以对准银河朝向）。
   - 双线性 + 方位 wrap、极点钳位。
   - 引力透镜自动生效：`render.py` 已按光线逃逸方向 `(th, ph)` 调用 `sky_color`，偏折后的方向自然采到被透镜的星图。
4. **程序化叠加**：保留现有程序化撒点/微光作为**可选叠加层**（`STARMAP_PROC_MIX`，默认弱），图像底图为主。

**新旋钮**（`config.py`）：

- `STARMAP_PATH`（None 则回退纯程序化，保持向后兼容）
- `STARMAP_RES`（默认 4096×2048；显存敏感，注释提示 8192×4096 的占用）
- `STARMAP_GAIN`（底图亮度增益）
- `STARMAP_ROT`（方位旋转，对准银河）+ `STARMAP_FLIP`（朝向翻转）
- `STARMAP_PROC_MIX`（程序化叠加权重，0=纯图，默认弱）

**显存注记**：`STARMAP_RES` RGB f32 的显存 = W×H×3×4 字节（4096×2048 ≈ 100MB，8192×4096 ≈ 400MB）。默认取 4096×2048；如紧张可在 loader 内降采样或后续考虑 f16。

**回归保证**：`STARMAP_PATH=None` 时回退当前纯程序化星空。

### D. 后期电影化 —— `postprocess.py` + `config.py`

**问题**：后期偏"渲染默认"，缺电影质感。

**改动**：

1. **Tonemap 升级**：在 `k_composite` 中加 config 可选的 filmic 曲线（ACES 或 AgX 近似），**现有自定义曲线保留为 `TONEMAP=0` 选项**，新曲线为高光滚降更柔的电影向选项。
2. **变形宽银幕 bloom 拉丝（anamorphic streak）**：新增一条横向偏置的宽模糊累加（复用/扩展 bloom 缓冲链，或单独一条横向 1D blur），产生科幻片水平镜头光条；强度/长度/色染旋钮控制，默认克制。
3. **收尾质感**（均在 composite kernel，均带旋钮，默认克制）：
   - **vignette** 暗角（基于像素到中心的归一化距离）。
   - **film grain** 胶片颗粒（基于像素坐标 + 帧相位的哈希，幅度旋钮；默认极弱或关）。
   - **chromatic aberration** 边缘色散（按到中心距离对 R/B 通道做微小径向偏移采样）。

**新旋钮**（`config.py`）：

- `TONEMAP`（0=现有曲线 / 1=ACES / 2=AgX）
- `ANAMORPHIC_ON/STRENGTH/LENGTH/TINT`
- `VIGNETTE_STRENGTH`
- `GRAIN_STRENGTH`（默认 0 或极弱）
- `CHROMA_AB_STRENGTH`（默认 0 或极弱）

**回归保证**：`TONEMAP=0`、anamorphic/vignette/grain/chroma 全 0 时，输出与当前一致。

## 跨工作线原则

- **逐项可关、默认安全**：每条线的全部旋钮取"原版/关闭"值时，渲染结果与当前完全一致。这保证增强是叠加式的，便于 A/B 对比与回退。
- **GUI 自动降级**：重开销项（A 的额外 octave、C 的高分底图、D 的 anamorphic）在 GUI 模式经 `ti.static` 标志或 `GUI_*` 覆盖降级。沿用 `config.py` 末尾 `if INTERACTIVE:` 的派生模式。
- **不改模块边界与函数签名约定**：新逻辑就地扩展现有 `@ti.func`/`@ti.kernel`，新增 `@ti.func` 辅助函数；`render.py` 积分循环仅新增 `r_min` 标量维护。

## 范围外（YAGNI）

- 物理建模（Novikov-Thorne 温度剖面、黑体光谱、光谱多普勒/引力红移配色）——方向选了电影观感，排除。
- 积分器/性能优化（RK45 七次求值、TAA 重跑测地线、噪声 octave 开销）——独立 spec。
- 色温大改（brainstorm 第 3 项未选）——仅靠现有 `DISK_TINT`/`OUTPUT_TINT` 做轻微冷暖。

## 验收

- 全旋钮取原版值 → 与当前输出逐像素一致（回归基线）。
- 逐条开启 A/B/C/D → 各自产生预期视觉变化且 GUI 档仍可实时。
- 出一张 4K 对比图（全开 vs 原版）确认整体电影观感提升。
