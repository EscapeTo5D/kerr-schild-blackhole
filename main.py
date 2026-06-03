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
    paused = False
    mouse_prev = None
    print("← → 转方位 | ↑ ↓ 仰角 | 空格 暂停 | R 复位 | P 截图 | ESC 退出")
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key == 'r':
                phi, theta, t = CAMERA_PHI, CAMERA_THETA, TIME_OFFSET
            elif e.key == 'p':
                ti.tools.imwrite(fields.disp, _rel("interstellar_ks_shot.png"))
                print("已截图: interstellar_ks_shot.png")
        if gui.is_pressed(ti.GUI.LEFT):
            phi -= 0.03
        if gui.is_pressed(ti.GUI.RIGHT):
            phi += 0.03
        if gui.is_pressed(ti.GUI.UP):
            theta = min(theta + 0.02, 1.4)
        if gui.is_pressed(ti.GUI.DOWN):
            theta = max(theta - 0.02, -1.4)
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if mouse_prev is not None:
                phi += (mx - mouse_prev[0]) * 3.0
                theta = min(max(theta + (my - mouse_prev[1]) * 3.0, -1.4), 1.4)
            mouse_prev = (mx, my)
        else:
            mouse_prev = None
        if not paused:
            t += 0.02
        render_frame(phi, theta, t, samples=1)
        gui.set_image(fields.disp)
        gui.text(f"a={SPIN}  phi={phi:.2f} theta={theta:.2f}" + ("  [PAUSED]" if paused else ""),
                 pos=(0.02, 0.97), color=0xFFFFFF)
        gui.show()


def main():
    fields.build()
    fields.load_noise_tex(NOISE_TEX_PATH or None)
    fields.load_organic_tex(ORGANIC_TEX_PATH or None)
    if INTERACTIVE:
        run_interactive()
    elif ANIMATE:
        render_sequence()
    else:
        save_image(out_path=OUT_PATH)


if __name__ == "__main__":
    main()
