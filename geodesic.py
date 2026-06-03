"""
Kerr-Schild 测地线引擎（移植自 blackhole_core.py，SPIN 取代 BH_SPIN）。

KS 笛卡尔 8 维相空间：get_metric / hamiltonianGradient / 自适应 RK45 / tetrad 相机标架。
位置天然是笛卡尔 (x,y,z)，盘在 z=0 面。
"""
import taichi as ti

from config import (vec3, vec4, mat4, SPIN,
                    RK_H_MAX, RK_H_MIN, RK_MAX_REJECT, RK_ATOL, RK_RTOL)


@ti.func
def diag(v: vec4) -> mat4:
    return mat4([[v.x, 0.0, 0.0, 0.0], [0.0, v.y, 0.0, 0.0],
                 [0.0, 0.0, v.z, 0.0], [0.0, 0.0, 0.0, v.w]])


@ti.func
def rFromCoords(pos: vec4) -> ti.f32:
    """KS 笛卡尔 → 克尔径向坐标 r（隐式方程解）。"""
    alpha = pos[1] ** 2 + pos[2] ** 2 + pos[3] ** 2 - SPIN ** 2
    r2 = 0.5 * (alpha + ti.sqrt(alpha ** 2 + 4.0 * SPIN ** 2 * pos[3] ** 2))
    return ti.sqrt(r2)


@ti.func
def kerrScalars(pos: vec4):
    r = rFromCoords(pos)
    xx, yy, zz = pos[1], pos[2], pos[3]
    denom = r * r + SPIN * SPIN
    k1 = (r * xx - SPIN * yy) / denom
    k2 = (r * yy + SPIN * xx) / denom
    k3 = zz / r
    f = 2.0 * r / (r * r + SPIN * SPIN * zz * zz / (r * r))
    return r, k1, k2, k3, f


@ti.func
def get_metric(pos: vec4) -> mat4:
    _r, k1, k2, k3, f = kerrScalars(pos)
    k = vec4(-1.0, k1, k2, k3)
    M = k.outer_product(k) * f
    M += diag(vec4(-1.0, 1.0, 1.0, 1.0))
    return M


@ti.func
def hamiltonianGradient(x: vec4, p: vec4) -> vec4:
    """∂H/∂x 解析梯度（KS 形式）。"""
    a = SPIN
    x1, x2, x3 = x[1], x[2], x[3]
    pt, px, py, pz = p[0], p[1], p[2], p[3]
    r, k1, k2, k3, f = kerrScalars(x)
    r2 = r * r
    a2 = a * a
    denom = r2 + a2
    r4 = r2 * r2
    a2z2 = a2 * x3 * x3
    fdenom = r4 + a2z2
    alpha = x1 * x1 + x2 * x2 + x3 * x3 - a2
    D = ti.sqrt(alpha * alpha + 4.0 * a2 * x3 * x3)
    lp = pt + k1 * px + k2 * py + k3 * pz

    dr_dx = r * x1 / D
    dr_dy = r * x2 / D
    dr_dz = denom * x3 / (r * D)

    df_dr = 2.0 * r2 * (3.0 * a2z2 - r4) / (fdenom * fdenom)
    df_dz_expl = -4.0 * a2 * x3 * r2 * r / (fdenom * fdenom)
    df_dx = df_dr * dr_dx
    df_dy = df_dr * dr_dy
    df_dz = df_dr * dr_dz + df_dz_expl

    dk1_dr = (a2 * x1 - r2 * x1 + 2.0 * r * a * x2) / (denom * denom)
    dk2_dr = (a2 * x2 - r2 * x2 - 2.0 * r * a * x1) / (denom * denom)
    dk3_dr = -x3 / r2

    dk1_dx = dk1_dr * dr_dx + r / denom
    dk1_dy = dk1_dr * dr_dy - a / denom
    dk1_dz = dk1_dr * dr_dz
    dk2_dx = dk2_dr * dr_dx + a / denom
    dk2_dy = dk2_dr * dr_dy + r / denom
    dk2_dz = dk2_dr * dr_dz
    dk3_dx = dk3_dr * dr_dx
    dk3_dy = dk3_dr * dr_dy
    dk3_dz = dk3_dr * dr_dz + 1.0 / r

    dlp_dx = dk1_dx * px + dk2_dx * py + dk3_dx * pz
    dlp_dy = dk1_dy * px + dk2_dy * py + dk3_dy * pz
    dlp_dz = dk1_dz * px + dk2_dz * py + dk3_dz * pz

    lp2 = lp * lp
    dH_dx = -0.5 * df_dx * lp2 - f * lp * dlp_dx
    dH_dy = -0.5 * df_dy * lp2 - f * lp * dlp_dy
    dH_dz = -0.5 * df_dz * lp2 - f * lp * dlp_dz
    return vec4(0.0, dH_dx, dH_dy, dH_dz)


@ti.func
def geodesicRHS(x: vec4, p: vec4):
    r, k1, k2, k3, f = kerrScalars(x)
    pt, px, py, pz = p[0], p[1], p[2], p[3]
    lp = pt + k1 * px + k2 * py + k3 * pz
    dx = vec4(-pt, px, py, pz) - f * lp * vec4(1.0, k1, k2, k3)
    dp = -hamiltonianGradient(x, p)
    return dx, dp


@ti.func
def transportStepRK45(x: vec4, p: vec4, h_in: ti.f32):
    """自适应 Dormand-Prince RK45 单步，返回 (x_new, p_new, h_used, h_next)。"""
    h_cur = ti.min(RK_H_MAX, ti.max(RK_H_MIN, h_in))
    x_new = x
    p_new = p
    h_used = h_cur
    h_next = h_cur
    reject = 0
    while reject < RK_MAX_REJECT:
        k1x, k1p = geodesicRHS(x, p)
        k2x, k2p = geodesicRHS(x + h_cur * (0.2 * k1x), p + h_cur * (0.2 * k1p))
        k3x, k3p = geodesicRHS(x + h_cur * (3.0 / 40.0 * k1x + 9.0 / 40.0 * k2x),
                               p + h_cur * (3.0 / 40.0 * k1p + 9.0 / 40.0 * k2p))
        k4x, k4p = geodesicRHS(
            x + h_cur * (44.0 / 45.0 * k1x - 56.0 / 15.0 * k2x + 32.0 / 9.0 * k3x),
            p + h_cur * (44.0 / 45.0 * k1p - 56.0 / 15.0 * k2p + 32.0 / 9.0 * k3p))
        k5x, k5p = geodesicRHS(
            x + h_cur * (19372.0 / 6561.0 * k1x - 25360.0 / 2187.0 * k2x
                         + 64448.0 / 6561.0 * k3x - 212.0 / 729.0 * k4x),
            p + h_cur * (19372.0 / 6561.0 * k1p - 25360.0 / 2187.0 * k2p
                         + 64448.0 / 6561.0 * k3p - 212.0 / 729.0 * k4p))
        k6x, k6p = geodesicRHS(
            x + h_cur * (9017.0 / 3168.0 * k1x - 355.0 / 33.0 * k2x
                         + 46732.0 / 5247.0 * k3x + 49.0 / 176.0 * k4x
                         - 5103.0 / 18656.0 * k5x),
            p + h_cur * (9017.0 / 3168.0 * k1p - 355.0 / 33.0 * k2p
                         + 46732.0 / 5247.0 * k3p + 49.0 / 176.0 * k4p
                         - 5103.0 / 18656.0 * k5p))
        x5 = x + h_cur * (35.0 / 384.0 * k1x + 500.0 / 1113.0 * k3x
                          + 125.0 / 192.0 * k4x - 2187.0 / 6784.0 * k5x + 11.0 / 84.0 * k6x)
        p5 = p + h_cur * (35.0 / 384.0 * k1p + 500.0 / 1113.0 * k3p
                          + 125.0 / 192.0 * k4p - 2187.0 / 6784.0 * k5p + 11.0 / 84.0 * k6p)
        k7x, k7p = geodesicRHS(x5, p5)
        ex = h_cur * (71.0 / 57600.0 * k1x - 71.0 / 16695.0 * k3x + 71.0 / 1920.0 * k4x
                      - 17253.0 / 339200.0 * k5x + 22.0 / 525.0 * k6x - 1.0 / 40.0 * k7x)
        ep = h_cur * (71.0 / 57600.0 * k1p - 71.0 / 16695.0 * k3p + 71.0 / 1920.0 * k4p
                      - 17253.0 / 339200.0 * k5p + 22.0 / 525.0 * k6p - 1.0 / 40.0 * k7p)
        acc = 0.0
        for c in ti.static(range(4)):
            scx = RK_ATOL + RK_RTOL * ti.max(ti.abs(x[c]), ti.abs(x5[c]))
            acc += (ex[c] / scx) ** 2
            scp = RK_ATOL + RK_RTOL * ti.max(ti.abs(p[c]), ti.abs(p5[c]))
            acc += (ep[c] / scp) ** 2
        err = ti.max(ti.sqrt(acc / 8.0), 1.0e-10)
        x_new = x5
        p_new = p5
        h_used = h_cur
        fac = ti.min(5.0, ti.max(0.2, 0.9 * err ** (-0.2)))
        h_next = ti.min(RK_H_MAX, ti.max(RK_H_MIN, h_cur * fac))
        if err <= 1.0 or h_cur <= RK_H_MIN:
            break
        h_cur = h_next
        reject += 1
    return x_new, p_new, h_used, h_next


@ti.func
def unit_vec(v: vec4, g: mat4) -> vec4:
    norm2 = (g @ v).dot(v)
    res = v
    if norm2 != 0.0:
        res = v / ti.sqrt(ti.abs(norm2))
    return res


@ti.func
def tetrad_basis(x: vec4, time_v: vec4, aim: vec4, vert: vec4) -> mat4:
    """局部正交标架（观察者参考系），返回每列一个基向量 (E0,E1,E2,E3)。"""
    g = get_metric(x)
    E0 = unit_vec(time_v, g)
    dot_aim_E0 = (g @ aim).dot(E0)
    E1 = unit_vec(aim + dot_aim_E0 * E0, g)
    dot_vert_E1 = (g @ vert).dot(E1)
    dot_vert_E0 = (g @ vert).dot(E0)
    E3 = unit_vec(vert - dot_vert_E1 * E1 + dot_vert_E0 * E0, g)
    E0_sp, E1_sp, E3_sp = vec3(E0[1], E0[2], E0[3]), vec3(E1[1], E1[2], E1[3]), vec3(E3[1], E3[2], E3[3])
    cross_13 = E1_sp.cross(E3_sp)
    comp0 = E0_sp.dot(cross_13)
    comp1 = -vec3(E0[2], E0[3], E0[0]).dot(vec3(E1[2], E1[3], E1[0]).cross(vec3(E3[2], E3[3], E3[0])))
    comp2 = vec3(E0[3], E0[0], E0[1]).dot(vec3(E1[3], E1[0], E1[1]).cross(vec3(E3[3], E3[0], E3[1])))
    comp3 = -vec3(E0[0], E0[1], E0[2]).dot(vec3(E1[0], E1[1], E1[2]).cross(vec3(E3[0], E3[1], E3[2])))
    E2_pre = vec4(comp0, comp1, comp2, comp3)
    E2 = unit_vec(g.inverse() @ E2_pre, g)
    return ti.Matrix.cols([E0, E1, E2, E3])
