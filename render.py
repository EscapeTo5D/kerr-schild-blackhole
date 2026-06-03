"""
渲染内核（KS 引擎 + 体积盘前到后合成 + TAA 抖动）+ 单帧渲染驱动。
"""
import math
import taichi as ti

from config import (vec3, vec4, CAM_DIST, SPIN, RES_X, RES_Y, FOV_FACTOR, INT_STEP,
                    MAX_STEPS, DISK_ON, DISK_RIN, DISK_ROUT, DISK_HALF_Z, DISK_STEP,
                    R_SKY, DISK_DOPPLER, DOPPLER_POWER, DOPPLER_TILT, SAMPLES,
                    CAMERA_PHI, CAMERA_THETA, TIME_OFFSET,
                    PHOTONRING_ON, PHOTONRING_STRENGTH, PHOTONRING_WIDTH, PHOTONRING_TINT, PHOTONRING_RADIUS)
import fields
from geodesic import tetrad_basis, get_metric, transportStepRK45, rFromCoords
from shading import gas_disc_sample, sky_color, haze_sample, disc_blueshift
from postprocess import k_clear, postprocess


@ti.kernel
def render(ang_phi: ti.f32, ang_theta: ti.f32, tparam: ti.f32, jx: ti.f32, jy: ti.f32, fov: ti.f32):
    # 相机位置（= blackhole_core 的 KS 嵌入）
    xy_proj = ti.sqrt(CAM_DIST ** 2 + SPIN ** 2) * ti.cos(ang_theta)
    cam_x = xy_proj * ti.cos(ang_phi)
    cam_y = xy_proj * ti.sin(ang_phi)
    cam_z = CAM_DIST * ti.sin(ang_theta)
    camPos = vec4(0.0, cam_x, cam_y, cam_z)

    time_v = vec4(1.0, 0.0, 0.0, 0.0)
    aim = vec4(0.0, cam_x, cam_y, cam_z)
    up_x = -ti.sqrt(CAM_DIST ** 2 + SPIN ** 2) * ti.sin(ang_theta) * ti.cos(ang_phi)
    up_y = -ti.sqrt(CAM_DIST ** 2 + SPIN ** 2) * ti.sin(ang_theta) * ti.sin(ang_phi)
    up_z = CAM_DIST * ti.cos(ang_theta)
    vert = vec4(0.0, up_x, up_y, up_z)
    axes = tetrad_basis(camPos, time_v, aim, vert)
    E0, E1, E2, E3 = axes[:, 0], axes[:, 1], axes[:, 2], axes[:, 3]

    rh = 1.0 + ti.sqrt(ti.max(1.0 - SPIN * SPIN, 0.0))

    # 多普勒强度的 cos²θ 仰角调制（= canvas 版 globalAlpha=cos²θ 思路）：edge-on 满、face-on 趋零
    ct = ti.cos(ang_theta)
    dopp_tilt = 1.0 - DOPPLER_TILT * (1.0 - ct * ct)

    for i, j in fields.pixels:
        ux = (2.0 * (i + 0.5 + jx) - RES_X) / RES_X
        uy = (2.0 * (j + 0.5 + jy) - RES_Y) / RES_X
        d_vec = vec3(-fov, ux, uy).normalized()
        dir4D = -E0 + d_vec.x * E1 + d_vec.y * E2 + d_vec.z * E3

        pos = camPos
        g_start = get_metric(pos)
        p = g_start @ dir4D

        col = vec3(0.0, 0.0, 0.0)
        alpha = 0.0
        h_step = INT_STEP
        last_pos = pos
        r_min = R_SKY
        for step in range(MAX_STEPS):
            # 接近赤道盘时按高度分级限步
            rho_cyl = ti.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
            az = ti.abs(pos[3])
            if DISK_ON == 1 and DISK_RIN - 2.0 < rho_cyl < DISK_ROUT + 2.0:
                if az < DISK_HALF_Z + 0.3:
                    h_step = ti.min(h_step, DISK_STEP)
                elif az < 2.0:
                    h_step = ti.min(h_step, 0.3)

            last_pos = pos
            pos, p, h_used, h_step = transportStepRK45(pos, p, h_step)
            r_curr = rFromCoords(pos)
            r_min = ti.min(r_min, r_curr)

            if r_curr < rh + 0.05:                        # 落入视界 → 其后黑
                break
            if r_curr > R_SKY:                            # 逃逸 → 背景星空透过 (1-α)
                dvec = vec3(pos[1], pos[2], pos[3]).normalized()
                th = ti.acos(ti.min(ti.max(dvec.z, -1.0), 1.0))
                ph = ti.atan2(dvec.y, dvec.x) % (2.0 * math.pi)
                col += (1.0 - alpha) * sky_color(th, ph)
                break
            if DISK_ON == 1:
                ds = ti.sqrt((pos[1] - last_pos[1]) ** 2 + (pos[2] - last_pos[2]) ** 2
                             + (pos[3] - last_pos[3]) ** 2)
                samp = gas_disc_sample(vec3(pos[1], pos[2], pos[3]), ds, tparam)
                emis = vec3(samp[0], samp[1], samp[2])
                cov = samp[3]
                if cov > 0.0:
                    if DISK_DOPPLER == 1:
                        b = pos[1] * p[2] - pos[2] * p[1]   # 轴向角动量 L_z = x·p_y − y·p_x
                        g = disc_blueshift(r_curr, b)
                        emis = emis * ti.pow(g, DOPPLER_POWER * dopp_tilt)
                    col += (1.0 - alpha) * emis * cov
                    alpha += (1.0 - alpha) * cov
                col += haze_sample(vec3(pos[1], pos[2], pos[3]), alpha, ds)
                if alpha > 0.995:
                    break
        if ti.static(PHOTONRING_ON == 1):
            r_ph = PHOTONRING_RADIUS
            if r_ph <= 0.0:
                r_ph = 2.0 * (1.0 + ti.cos(2.0 / 3.0 * ti.acos(-SPIN)))  # 顺行赤道光子轨道半径
            dr = (r_min - r_ph) / PHOTONRING_WIDTH
            ring = ti.exp(-dr * dr) * PHOTONRING_STRENGTH
            col += (1.0 - alpha) * ring * vec3(PHOTONRING_TINT[0],
                                               PHOTONRING_TINT[1], PHOTONRING_TINT[2])
        fields.samp_acc[i, j] += col


@ti.kernel
def k_avg_to_pixels(inv_n: ti.f32):
    for i, j in fields.pixels:
        fields.pixels[i, j] = fields.samp_acc[i, j] * inv_n


def _halton(idx, base):
    f = 1.0
    r = 0.0
    while idx > 0:
        f /= base
        r += f * (idx % base)
        idx //= base
    return r


def render_frame(phi=CAMERA_PHI, theta=CAMERA_THETA, t=TIME_OFFSET, samples=None, fov=FOV_FACTOR):
    n = SAMPLES if samples is None else samples
    k_clear(fields.samp_acc)
    for s in range(n):
        jx = _halton(s + 1, 2) - 0.5 if n > 1 else 0.0
        jy = _halton(s + 1, 3) - 0.5 if n > 1 else 0.0
        render(phi, theta, t, jx, jy, fov)
    k_avg_to_pixels(1.0 / n)
    postprocess()
    ti.sync()
