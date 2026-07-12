import taichi as ti
import math

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS.")

# --- Physics Constants ---
N        = 150000
G        = 1.0
EPSILON  = 1e-3

# Dynamic Black Hole fields
M_BH             = ti.field(dtype=ti.f32, shape=())
r_s              = ti.field(dtype=ti.f32, shape=())
particles_consumed = ti.field(dtype=ti.i32, shape=())
quasar_intensity = ti.field(dtype=ti.f32, shape=())

# Simulation state
dt_base = 0.002
dt      = ti.field(dtype=ti.f32, shape=())

# Particle fields
pos   = ti.Vector.field(3, dtype=ti.f32, shape=N)
vel   = ti.Vector.field(3, dtype=ti.f32, shape=N)
color = ti.Vector.field(3, dtype=ti.f32, shape=N)
mass  = ti.field(dtype=ti.f32, shape=N)
life  = ti.field(dtype=ti.f32, shape=N)

# Render fields
render_pos   = ti.Vector.field(3, dtype=ti.f32, shape=N)
render_color = ti.Vector.field(3, dtype=ti.f32, shape=N)

# Black hole render
bh_pos   = ti.Vector.field(3, dtype=ti.f32, shape=1)
bh_color = ti.Vector.field(3, dtype=ti.f32, shape=1)

# Camera
cam_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

# Mouse
mouse_pos3d = ti.Vector.field(3, dtype=ti.f32, shape=())
mouse_active = ti.field(dtype=ti.i32, shape=())

# -------------------------------------------------------
@ti.func
def rand3d():
    return ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5]) * 2.0

# -------------------------------------------------------
@ti.func
def respawn_particle(i):
    rs  = r_s[None]
    mbh = M_BH[None]

    if quasar_intensity[None] > ti.random() * 2.5:   # raised threshold: jets never steal 100% of particles
        # ---- Quasar jet (helix structure) ----
        side      = 1.0 if ti.random() > 0.5 else -1.0
        bh_scale  = rs / 2.5
        base_angle = ti.random() * math.pi * 2.0

        # Distribute spawn point along the helix height
        t          = ti.random()                              # 0..1 along jet
        jet_len    = rs * (12.0 + 8.0 * ti.min(bh_scale, 2.0))
        y_pos      = side * (rs * 1.1 + t * jet_len)

        # Helix: angle winds 3 full turns over jet length
        turns      = 3.0
        h_angle    = base_angle + turns * 2.0 * math.pi * t
        h_radius   = rs * (0.35 + 0.2 * ti.min(bh_scale, 3.0))

        pos[i] = ti.Vector([h_radius * ti.cos(h_angle), y_pos, h_radius * ti.sin(h_angle)])

        # Velocity: axial (escape) + tangential (follows helix winding)
        r_start    = rs * 1.1
        v_esc      = ti.sqrt(2.0 * G * mbh / (r_start + EPSILON))
        v_axial    = side * v_esc * (1.8 + ti.random() * 1.0)
        omega_h    = turns * 2.0 * math.pi / (jet_len + EPSILON)
        v_tan      = h_radius * omega_h * ti.abs(v_axial)
        v_x        = -ti.sin(h_angle) * v_tan
        v_z        =  ti.cos(h_angle) * v_tan

        vel[i]   = ti.Vector([v_x, v_axial, v_z])
        mass[i]  = 0.5
        life[i]  = 1.0
        color[i] = ti.Vector([0.6, 0.2, 1.0]) * 5.0
    else:
        # ---- Accretion disk ----
        angle = ti.random() * math.pi * 2.0
        # Inner visible region: 1.5–12× r_s (keeps disk on-screen and bright)
        r     = rs * 1.5 + ti.random() * rs * 10.5

        # Thin flat disk
        y_off = (ti.random() - 0.5) * 0.6

        p      = ti.Vector([r * ti.cos(angle), y_off, r * ti.sin(angle)])
        pos[i] = p

        v_mag  = ti.sqrt(G * mbh * r / ((r - rs + EPSILON) ** 2))
        tangent = ti.Vector([-ti.sin(angle), 0.0, ti.cos(angle)])
        radial  = ti.Vector([-ti.cos(angle), 0.0, -ti.sin(angle)])
        vel[i]  = tangent * v_mag + radial * (v_mag * 0.05) + rand3d() * (v_mag * 0.02)

        mass[i] = 1.0 + ti.random() * 2.0
        life[i] = 1.0

        # Temperature → colour (kinetic energy proxy) — boosted for visibility
        speed_sq = vel[i].norm_sqr()
        T_norm   = ti.min(1.0, speed_sq / 1600.0)
        c_cold = ti.Vector([0.1,  0.5,  2.0])   # bright blue
        c_mid  = ti.Vector([1.5,  0.7,  0.1])   # bright orange
        c_hot  = ti.Vector([3.0,  0.3,  0.0])   # very bright red-orange
        if T_norm < 0.5:
            f = T_norm * 2.0
            color[i] = c_cold * (1.0 - f) + c_mid * f
        else:
            f = (T_norm - 0.5) * 2.0
            color[i] = c_mid * (1.0 - f) + c_hot * f
        color[i] += rand3d() * 0.15

# -------------------------------------------------------
@ti.kernel
def init_particles():
    dt[None]     = dt_base
    # Seed BH mass large enough to capture collapsing core (~50 r_s initial disk)
    M_BH[None]   = 5000.0
    r_s[None]    = 2.5          # 5000 * 0.0005
    quasar_intensity[None]  = 0.0
    particles_consumed[None] = 0
    bh_pos[0]   = ti.Vector([0.0, 0.0, 0.0])
    bh_color[0] = ti.Vector([0.02, 0.0, 0.0])

    for i in range(N):
        # Progenitor star: uniform sphere, radius 5–30
        radius = 5.0 + ti.random() * 25.0
        theta  = ti.random() * math.pi * 2.0
        phi    = ti.acos(2.0 * ti.random() - 1.0)

        px = radius * ti.sin(phi) * ti.cos(theta)
        py = radius * ti.cos(phi)
        pz = radius * ti.sin(phi) * ti.sin(theta)
        p  = ti.Vector([px, py, pz])
        pos[i] = p

        # Slow solid-body rotation (hydrostatic star)
        omega  = 0.08
        vel[i] = ti.Vector([-omega * py, omega * px, 0.0]) * 0.3

        mass[i] = 1.0 + ti.random() * 2.0
        life[i] = 1.0
        # Pre-supernova star colour: bright orange-white
        color[i] = ti.Vector([1.0, 0.85, 0.6]) * (1.5 + ti.random())

# -------------------------------------------------------
@ti.kernel
def trigger_supernova():
    """
    Core-collapse supernova.
    Split star into two zones:
      - Inner core  (r < collapse_r): free-fall inward → feeds BH immediately
      - Outer shell (collapse_r ≤ r ≤ star_r): moderate outward blast,
        CAPPED so particles remain gravitationally bound and fall back in
    """
    mbh        = M_BH[None]
    collapse_r = 10.0   # inner core radius
    star_r     = 30.0   # outer edge of star

    for i in range(N):
        p      = pos[i]
        radius = p.norm() + EPSILON
        p_hat  = p / radius

        if radius <= collapse_r:
            # Free-fall: give inward velocity ≈ freefall speed at that radius
            v_ff  = ti.sqrt(2.0 * G * mbh / radius)
            v_ff  = ti.min(v_ff, 20.0)                  # safety cap (reduced intensity)
            vel[i] = p_hat * (-v_ff) + rand3d() * 0.5  # inward

        elif radius <= star_r:
            # Outer envelope: gentle outward kick, then hard-cap total speed
            # so every particle stays gravitationally bound and falls back.
            v_esc     = ti.sqrt(2.0 * G * mbh / radius)
            frac      = (radius - collapse_r) / (star_r - collapse_r)
            v_blast   = frac * v_esc * 0.18              # 18 % radial kick
            v_blast  += (ti.random() - 0.5) * 0.3
            vel[i]   += p_hat * v_blast

            # Hard cap: total speed ≤ 60 % of escape velocity → always bound
            v_cap  = v_esc * 0.60
            speed  = vel[i].norm()
            if speed > v_cap:
                vel[i] = vel[i] * (v_cap / (speed + EPSILON))

        # Particles outside star_r are untouched

# -------------------------------------------------------
@ti.kernel
def spawn_at_mouse():
    if mouse_active[None] == 1:
        rs  = r_s[None]
        mbh = M_BH[None]
        for _ in range(100):
            idx   = int(ti.random() * N)
            m_pos = mouse_pos3d[None]
            p     = m_pos + rand3d() * 2.0
            pos[idx] = p

            r      = p.norm() + EPSILON
            v_mag  = ti.sqrt(G * mbh * r / ((r - rs + EPSILON) ** 2))

            angle  = ti.atan2(p.z, p.x)
            tangent = ti.Vector([-ti.sin(angle), 0.0, ti.cos(angle)])
            vel[idx]   = tangent * v_mag
            mass[idx]  = 1.0 + ti.random() * 2.0
            life[idx]  = 1.0
            color[idx] = ti.Vector([1.0, 0.9, 0.6])

# -------------------------------------------------------
@ti.kernel
def feed_quasar():
    """Periodically inject new jet particles when quasar is active."""
    qi  = quasar_intensity[None]
    rs  = r_s[None]
    mbh = M_BH[None]
    if qi > 0.1:
        # Particle count and jet base scale proportionally with BH size
        bh_scale  = rs / 2.5
        jet_count = int(qi * 800 * ti.min(bh_scale, 4.0))
        for _ in range(jet_count):
            idx        = int(ti.random() * N)
            side       = 1.0 if ti.random() > 0.5 else -1.0
            base_angle = ti.random() * math.pi * 2.0

            # Helix position: distribute along jet height
            t        = ti.random()
            jet_len  = rs * (12.0 + 8.0 * ti.min(bh_scale, 2.0))
            y_pos    = side * (rs * 1.05 + t * jet_len)
            turns    = 3.0
            h_angle  = base_angle + turns * 2.0 * math.pi * t
            h_radius = rs * (0.35 + 0.2 * ti.min(bh_scale, 3.0))

            pos[idx] = ti.Vector([h_radius * ti.cos(h_angle), y_pos, h_radius * ti.sin(h_angle)])

            r_start  = rs * 1.05
            v_esc    = ti.sqrt(2.0 * G * mbh / (r_start + EPSILON))
            v_axial  = side * v_esc * (1.5 + ti.random() * 1.0)
            omega_h  = turns * 2.0 * math.pi / (jet_len + EPSILON)
            v_tan    = h_radius * omega_h * ti.abs(v_axial)
            vel[idx] = ti.Vector([-ti.sin(h_angle) * v_tan, v_axial, ti.cos(h_angle) * v_tan])

            color[idx] = ti.Vector([1.5, 0.2, 2.5])
            mass[idx]  = 0.3
            life[idx]  = 1.0

# -------------------------------------------------------
@ti.kernel
def seed_disk():
    """Inject fresh accretion-disk particles every frame so the disk
    remains visible regardless of quasar state."""
    rs  = r_s[None]
    mbh = M_BH[None]
    for _ in range(400):
        idx   = int(ti.random() * N)
        angle = ti.random() * math.pi * 2.0
        # Hot inner disk: 1.5 to 12× r_s
        r     = rs * 1.5 + ti.random() * rs * 10.5
        y_off = (ti.random() - 0.5) * 0.5      # very thin flat disk

        pos[idx] = ti.Vector([r * ti.cos(angle), y_off, r * ti.sin(angle)])

        v_mag   = ti.sqrt(G * mbh * r / ((r - rs + EPSILON) ** 2))
        tangent = ti.Vector([-ti.sin(angle), 0.0, ti.cos(angle)])
        radial  = ti.Vector([-ti.cos(angle), 0.0, -ti.sin(angle)])
        vel[idx] = tangent * v_mag + radial * (v_mag * 0.05) + rand3d() * (v_mag * 0.01)

        mass[idx] = 1.0 + ti.random() * 2.0
        life[idx] = 1.0

        # Temperature → colour
        speed_sq = vel[idx].norm_sqr()
        T_norm   = ti.min(1.0, speed_sq / 1600.0)
        c_cold = ti.Vector([0.1,  0.5,  2.0])
        c_mid  = ti.Vector([1.5,  0.7,  0.1])
        c_hot  = ti.Vector([3.0,  0.3,  0.0])
        if T_norm < 0.5:
            f = T_norm * 2.0
            color[idx] = c_cold * (1.0 - f) + c_mid * f
        else:
            f = (T_norm - 0.5) * 2.0
            color[idx] = c_mid * (1.0 - f) + c_hot * f

# -------------------------------------------------------
@ti.kernel
def update(current_time: ti.f32):
    rs  = r_s[None]
    mbh = M_BH[None]

    for i in range(N):
        # During the first 1-second "star" phase: just display, don't integrate
        if current_time < 1.0:
            pos[i] = pos[i]
            continue

        p = pos[i]
        v = vel[i]

        r_vec = -p
        r     = r_vec.norm() + EPSILON
        r_dir = r_vec / r

        # Paczyński-Wiita pseudo-Newtonian gravity (clamp near horizon)
        r_eff     = ti.max(r - rs, 0.2)
        f_gravity = r_dir * (G * mbh / (r_eff ** 2))

        # Frame-drag (prograde spin)
        spin_dir = ti.Vector([-p.z, 0.0, p.x]) / r
        f_drag   = spin_dir * (G * mbh / (r ** 2 + EPSILON)) * 5.0

        f_total = f_gravity + f_drag

        v += f_total * dt[None]
        v *= (1.0 - 0.01 * dt[None])   # tiny dissipation
        p += v * dt[None]

        if r < rs:
            ti.atomic_add(M_BH[None], mass[i] * 0.1)   # absorb only 10% of mass → slower BH growth
            ti.atomic_add(particles_consumed[None], 1)
            respawn_particle(i)
        elif r > 300.0:
            respawn_particle(i)
        else:
            pos[i] = p
            vel[i] = v

            # Temperature colour (kinetic energy) — boosted to match spawn colours
            speed_sq = v.norm_sqr()
            T_norm   = ti.min(1.0, speed_sq / 1600.0)
            c_cold = ti.Vector([0.1,  0.5,  2.0])
            c_mid  = ti.Vector([1.5,  0.7,  0.1])
            c_hot  = ti.Vector([3.0,  0.3,  0.0])
            target = ti.Vector([0.0, 0.0, 0.0])
            if T_norm < 0.5:
                f = T_norm * 2.0
                target = c_cold * (1.0 - f) + c_mid * f
            else:
                f = (T_norm - 0.5) * 2.0
                target = c_mid * (1.0 - f) + c_hot * f
            color[i] = color[i] * 0.95 + target * 0.05

# -------------------------------------------------------
@ti.kernel
def compute_lensing():
    c_dir = cam_dir[None]
    rs    = r_s[None]

    bh_pos[0]   = ti.Vector([0.0, 0.0, 0.0])
    bh_color[0] = ti.Vector([0.0, 0.0, 0.0])

    for i in range(N):
        p = pos[i]
        c = color[i]

        z_c    = p.dot(c_dir)
        p_proj = p - z_c * c_dir
        d      = p_proj.norm()

        if z_c < 0.0:
            if d > 0.01:
                deflection  = (rs * rs * 1.5) / (d + EPSILON)
                d_new       = d + deflection
                p_proj_new  = p_proj * (d_new / d)
                p           = p_proj_new + z_c * c_dir
                if d_new < rs * 2.6:
                    c = ti.Vector([0.0, 0.0, 0.0])
        else:
            if p.norm() < rs:
                c = ti.Vector([0.0, 0.0, 0.0])

        render_pos[i]   = p
        render_color[i] = c

# -------------------------------------------------------
def main():
    init_particles()

    window = ti.ui.Window("Black Hole Simulator", (1280, 800), vsync=True)
    canvas = window.get_canvas()
    scene  = ti.ui.Scene()
    camera = ti.ui.Camera()

    cam_theta  = 0.0
    cam_phi    = 0.4
    cam_radius = 120.0

    paused               = False
    speed_mult           = 1.0
    global_accretion_ema = 0.0
    sim_time             = 0.0
    supernova_done       = False

    print("\n" + "="*40)
    print("BLACK HOLE SIMULATION CONTROLS")
    print("SPACE : Pause/Resume")
    print("UP    : Speed up time")
    print("DOWN  : Slow down time")
    print("W / S : Zoom in / out")
    print("R     : Reset simulation")
    print("LMB   : Spawn particles at cursor")
    print("="*40 + "\n")

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key in ('r', 'R'):
                init_particles()
                sim_time       = 0.0
                supernova_done = False
            elif e.key == ti.ui.UP:
                speed_mult *= 1.5
            elif e.key == ti.ui.DOWN:
                speed_mult /= 1.5

        if window.is_pressed('w') or window.is_pressed('W'):
            cam_radius = max(20.0, cam_radius - 0.5)
        if window.is_pressed('s') or window.is_pressed('S'):
            cam_radius = min(400.0, cam_radius + 0.5)

        # Mouse → world position on disk plane y=0
        mouse_active[None] = 0
        if window.is_pressed(ti.ui.LMB):
            mouse_active[None] = 1
            mx, my = window.get_cursor_pos()

            cx = cam_radius * math.cos(cam_phi) * math.sin(cam_theta)
            cy = cam_radius * math.sin(cam_phi)
            cz = cam_radius * math.cos(cam_phi) * math.cos(cam_theta)

            lC   = math.sqrt(cx*cx + cy*cy + cz*cz)
            Fx, Fy, Fz = -cx/lC, -cy/lC, -cz/lC

            lR   = math.sqrt(Fz*Fz + Fx*Fx)
            Rx, Ry, Rz = -Fz/lR, 0.0, Fx/lR

            Vx = Ry*Fz - Rz*Fy
            Vy = Rz*Fx - Rx*Fz
            Vz = Rx*Fy - Ry*Fx
            lV = math.sqrt(Vx*Vx + Vy*Vy + Vz*Vz)
            Vx, Vy, Vz = Vx/lV, Vy/lV, Vz/lV

            tan_y = math.tan(math.radians(30))
            tan_x = tan_y * 1.6

            ndc_x = 2.0*mx - 1.0
            ndc_y = 2.0*my - 1.0

            Dx = Fx + ndc_x*tan_x*Rx + ndc_y*tan_y*Vx
            Dy = Fy + ndc_x*tan_x*Ry + ndc_y*tan_y*Vy
            Dz = Fz + ndc_x*tan_x*Rz + ndc_y*tan_y*Vz

            if abs(Dy) > 1e-5:
                t       = -cy / Dy
                world_x = cx + t*Dx
                world_z = cz + t*Dz
            else:
                world_x, world_z = 0.0, 0.0

            mouse_pos3d[None] = ti.Vector([world_x, 0.0, world_z])

        if not paused:
            dt[None]  = dt_base * speed_mult
            delta_t   = dt[None] * 3.0

            # Fire supernova once at t≈1 s
            if not supernova_done and sim_time < 1.0 and (sim_time + delta_t) >= 1.0:
                trigger_supernova()
                supernova_done = True

            sim_time += delta_t

            spawn_at_mouse()
            for _ in range(3):
                update(sim_time)
            feed_quasar()
            seed_disk()    # always maintain a visible accretion disk

            consumed                  = particles_consumed[None]
            particles_consumed[None]  = 0

            global_accretion_ema = global_accretion_ema * 0.95 + consumed * 0.05
            if global_accretion_ema > 0.5:
                quasar_intensity[None] = min(1.0, quasar_intensity[None] + 0.08)   # faster ramp (was +0.05)
            else:
                quasar_intensity[None] = max(0.0, quasar_intensity[None] - 0.01)   # slower decay (was -0.02)

            # BH growth: rs = M * 0.0005, capped so it never fills the screen
            r_s[None]  = min(M_BH[None] * 0.0005, 12.0)
            cam_theta += 0.002

        cx = cam_radius * math.cos(cam_phi) * math.sin(cam_theta)
        cy = cam_radius * math.sin(cam_phi)
        cz = cam_radius * math.cos(cam_phi) * math.cos(cam_theta)

        camera.position(cx, cy, cz)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)

        cam_dir[None] = ti.Vector([cx, cy, cz]).normalized()
        compute_lensing()

        scene.set_camera(camera)
        scene.ambient_light((0.1, 0.1, 0.15))
        scene.point_light(pos=(0.0, 0.0, 0.0), color=(1.0, 0.8, 1.0))

        scene.particles(render_pos, per_vertex_color=render_color, radius=0.04)

        current_rs = r_s[None]
        scene.particles(bh_pos, per_vertex_color=bh_color, radius=current_rs)

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()
