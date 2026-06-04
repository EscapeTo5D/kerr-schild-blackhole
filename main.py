"""
interstellar_ks · 入口。

在 IDE 里直接 Run 本文件即可（参数在 config.py 里改）。
纹理等相对路径由 config._rel 锚定到 PROJECT_ROOT，与 IDE 的工作目录无关。
"""
import taichi as ti

import math
import os

from config import (SPIN, RES_X, RES_Y, SAMPLES, CAMERA_PHI, CAMERA_THETA, TIME_OFFSET,
                    OUT_PATH, NOISE_TEX_PATH, ORGANIC_TEX_PATH, INTERACTIVE, _rel,
                    FOV_FACTOR, GUI_TIME_STEP, GUI_ZOOM_STEP,
                    CAMERA_ROLL, CAMERA_PAN_X, CAMERA_PAN_Y, CAM_DIST,
                    ANIMATE, ANIM_FRAMES, ANIM_FPS, ANIM_CAM_TURNS, ANIM_DISK_TURN, ANIM_OUT_DIR,
                    ANIM_AUTO_MERGE, ANIM_VIDEO)
import fields
from render import render_frame


def save_image(out_path=OUT_PATH, t=TIME_OFFSET):
    import time
    print(f"渲染 KS 黑洞 (a={SPIN}, RES={RES_X}x{RES_Y}, SAMPLES={SAMPLES}) ...")
    render_frame(CAMERA_PHI, CAMERA_THETA, t, samples=1)   # 预热（JIT）
    start = time.time()
    render_frame(CAMERA_PHI, CAMERA_THETA, t)
    print(f"渲染耗时(纯渲染): {time.time() - start:.3f}s")
    ti.tools.imwrite(fields.disp, out_path)
    print(f"已保存: {out_path}")


def _merge_video():
    """用 ffmpeg 把 frames_ks/ 的序列帧合成 mp4；没装 ffmpeg 则回退为打印命令。"""
    import shutil
    import subprocess
    pattern = os.path.join(ANIM_OUT_DIR, "frame_%04d.png")
    cmd = ["ffmpeg", "-y", "-framerate", str(ANIM_FPS), "-i", pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", ANIM_VIDEO]
    if shutil.which("ffmpeg") is None:
        print("未找到 ffmpeg（不在 PATH）。装好后手动执行：")
        print("  " + " ".join(f'"{a}"' if " " in a else a for a in cmd))
        return
    print("合成视频中 ...")
    r = subprocess.run(cmd)
    if r.returncode == 0:
        print(f"已合成: {ANIM_VIDEO}")
    else:
        print(f"ffmpeg 合成失败（返回码 {r.returncode}，见上方输出）。")


def render_sequence():
    """序列帧动画：每帧推进相机方位 phi（环绕）和噪声相位 t（盘流动），存编号图。"""
    os.makedirs(ANIM_OUT_DIR, exist_ok=True)
    print(f"序列帧 {ANIM_FRAMES} 帧 @ {RES_X}x{RES_Y}, SAMPLES={SAMPLES} → {ANIM_OUT_DIR}")
    render_frame(CAMERA_PHI, CAMERA_THETA, TIME_OFFSET, samples=1)   # 预热（JIT）
    for f in range(ANIM_FRAMES):
        frac = f / ANIM_FRAMES                                        # 0..1（首尾不重复，便于循环播放）
        phi = CAMERA_PHI + ANIM_CAM_TURNS * 2.0 * math.pi * frac
        t = TIME_OFFSET + ANIM_DISK_TURN * frac
        render_frame(phi, CAMERA_THETA, t)
        path = os.path.join(ANIM_OUT_DIR, f"frame_{f:04d}.png")
        ti.tools.imwrite(fields.disp, path)
        print(f"  [{f + 1}/{ANIM_FRAMES}] {path}")
    print(f"渲染完成。时长 {ANIM_FRAMES / ANIM_FPS:.1f}s @ {ANIM_FPS}fps。")
    if ANIM_AUTO_MERGE:
        _merge_video()
    else:
        print("合成视频（在项目根执行）：")
        print(f'  ffmpeg -framerate {ANIM_FPS} -i "{ANIM_OUT_DIR}/frame_%04d.png" '
              f'-c:v libx264 -pix_fmt yuv420p -crf 18 "{ANIM_VIDEO}"')


def run_interactive():
    gui = ti.GUI("Interstellar · Kerr-Schild", res=(RES_X, RES_Y))
    phi, theta, t = CAMERA_PHI, CAMERA_THETA, TIME_OFFSET
    fov = FOV_FACTOR
    roll, pan_x, pan_y, cam_dist = CAMERA_ROLL, CAMERA_PAN_X, CAMERA_PAN_Y, CAM_DIST
    paused = False
    mouse_prev = None
    print("【相机实时调参】")
    print("  ← → 转方位 phi | ↑ ↓ 仰角 theta | 鼠标左键拖拽 = phi/theta")
    print("  Q E 滚转 roll | A D 主点左右 pan_x | W S 主点上下 pan_y")
    print("  滚轮 推拉远近 cam_dist | Ctrl+滚轮 缩放视角 fov")
    print("  空格 暂停盘流动 | R 复位 | P 截图 | ESC 退出")
    print("  调好后看左上角数值，把 ROLL/PAN_X/PAN_Y/THETA/FOV_FACTOR/CAM_DIST 填回 config.py")
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS, ti.GUI.WHEEL):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key == 'r':
                phi, theta, t, fov = CAMERA_PHI, CAMERA_THETA, TIME_OFFSET, FOV_FACTOR
                roll, pan_x, pan_y, cam_dist = CAMERA_ROLL, CAMERA_PAN_X, CAMERA_PAN_Y, CAM_DIST
            elif e.key == 'p':
                ti.tools.imwrite(fields.disp, _rel("interstellar_ks_shot.png"))
                print(f"已截图: interstellar_ks_shot.png  | 当前相机: "
                      f"CAMERA_ROLL={roll:.3f} CAMERA_PAN_X={pan_x:.3f} CAMERA_PAN_Y={pan_y:.3f} "
                      f"CAMERA_THETA={theta:.3f} FOV_FACTOR={fov:.3f} CAM_DIST={cam_dist:.2f} CAMERA_PHI={phi:.3f}")
            elif e.key == ti.GUI.WHEEL:
                if gui.is_pressed(ti.GUI.CTRL):
                    # Ctrl+滚轮：缩放视角（FOV 变窄=放大）；相机位置不动
                    fov *= GUI_ZOOM_STEP if e.delta[1] > 0 else (1.0 / GUI_ZOOM_STEP)
                    fov = min(max(fov, 0.3), 8.0)
                else:
                    # 裸滚轮：推拉相机远近（上滚拉近）；几何随距离变（前景沉浸感）
                    cam_dist *= (1.0 / 1.06) if e.delta[1] > 0 else 1.06
                    cam_dist = min(max(cam_dist, 5.0), 120.0)
        if gui.is_pressed(ti.GUI.LEFT):
            phi -= 0.03
        if gui.is_pressed(ti.GUI.RIGHT):
            phi += 0.03
        if gui.is_pressed(ti.GUI.UP):
            theta = min(theta + 0.02, 1.4)
        if gui.is_pressed(ti.GUI.DOWN):
            theta = max(theta - 0.02, -1.4)
        if gui.is_pressed('q'):
            roll -= 0.01
        if gui.is_pressed('e'):
            roll += 0.01
        if gui.is_pressed('a'):
            pan_x -= 0.01
        if gui.is_pressed('d'):
            pan_x += 0.01
        if gui.is_pressed('w'):
            pan_y += 0.01
        if gui.is_pressed('s'):
            pan_y -= 0.01
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if mouse_prev is not None:
                phi += (mx - mouse_prev[0]) * 3.0
                theta = min(max(theta + (my - mouse_prev[1]) * 3.0, -1.4), 1.4)
            mouse_prev = (mx, my)
        else:
            mouse_prev = None
        if not paused:
            t += GUI_TIME_STEP
        render_frame(phi, theta, t, fov=fov, roll=roll, pan_x=pan_x, pan_y=pan_y, cam_dist=cam_dist)
        gui.set_image(fields.disp)
        gui.text(f"a={SPIN}  phi={phi:.2f} theta={theta:.3f} fov={fov:.2f} cam_dist={cam_dist:.1f}"
                 + ("  [PAUSED]" if paused else ""),
                 pos=(0.02, 0.97), color=0xFFFFFF)
        gui.text(f"roll={roll:.3f}  pan_x={pan_x:.3f}  pan_y={pan_y:.3f}",
                 pos=(0.02, 0.94), color=0xFFFF66)
        gui.show()


def main():
    fields.build()
    fields.load_noise_tex(NOISE_TEX_PATH or None)
    fields.load_organic_tex(ORGANIC_TEX_PATH or None)
    from config import STARMAP_ON, STARMAP_PATH
    if STARMAP_ON == 1:                        # 开关+路径都满足才加载（关时 starmap_tex 是 (1,1) 占位，加载会崩）
        fields.load_starmap_tex(STARMAP_PATH)
    if INTERACTIVE:
        run_interactive()
    elif ANIMATE:
        render_sequence()
    else:
        save_image(out_path=OUT_PATH)


if __name__ == "__main__":
    main()
