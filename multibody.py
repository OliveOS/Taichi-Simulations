import taichi as ti

#init taichi context + window details
ti.init(arch=ti.gpu) #excepthook property seems broken on mac?
n: int = 1200
pixels = ti.field(dtype=float, shape=(n, n))

#setup body details and properties
nbodies = 2
body_field = ti.Struct.field({
    "pos": ti.types.vector(2, ti.f32),
    "vel": ti.types.vector(2, ti.f32),
    "acc": ti.types.vector(2, ti.f32),
    "mass": ti.f32,
}, shape=(nbodies,))

body_field[0].mass = 2
body_field[0].pos = ti.Vector([0.5, 0.3])
body_field[0].vel = ti.Vector([0, 4.5e-6])
body_field[0].acc = ti.Vector([0.0, 0.0])
body_field[1].mass = 1
body_field[1].pos = ti.Vector([0.2, 0.3])
body_field[1].vel = ti.Vector([0, -3.14e-6])
body_field[1].acc = ti.Vector([0.0, 0.0])

#time-step details
h = 10**(-5)
substepping = 10

#normalizes a ti vector
@ti.func
def normalize2D(vec):
    length = ti.sqrt(vec[0]**2 + vec[1]**2)
    return ti.Vector([vec[0]/length, vec[1]/length])

#returns the gravitational force vector between two bodies
@ti.func
def grav_force(m1, m2):
    G = 6.67 * 10**(-11)
    r = ti.sqrt((m1.pos[0]-m2.pos[0])**2 + (m1.pos[1]-m2.pos[1])**2)
    return -G * m1.mass * m2.mass * 1/r**2 * normalize2D(ti.Vector([m1.pos[0]-m2.pos[0], m1.pos[1]-m2.pos[1]]))

#parallelized taichi function to update body accelerations
@ti.func
def update_accs(bodies) -> None:
    for i in range(nbodies):
        bodies[i].acc = ti.Vector([0.0, 0.0])
        for j in range(nbodies):
            if i != j:
                bodies[i].acc += grav_force(bodies[i], bodies[j])/bodies[i].mass

@ti.func
def update_vels(bodies, dt: float) -> None:
    for i in range(nbodies):
        bodies[i].vel += bodies[i].acc * dt

@ti.func
def update_poss(bodies, dt: float) -> None:
    for i in range(nbodies):
        bodies[i].pos += bodies[i].vel * dt

@ti.kernel 
def update(dt: float):
    update_accs(body_field)
    update_vels(body_field, dt)
    update_poss(body_field, dt)

gui = ti.GUI("n-Body", res = (n, n))
accel_scale = 2e7
veloc_scale = 7.5e3

for i in range(100000):
    update(100)
    for i in range(nbodies):
        gui.circle(body_field[i].pos, radius = 15, color=0x068587)

    #acceleration vectors
    gui.arrow(body_field[0].pos, ti.Vector([body_field[0].acc[0] * accel_scale, body_field[0].acc[1] * accel_scale]), color = 0x4EA4EE)
    gui.arrow(body_field[1].pos, ti.Vector([body_field[1].acc[0] * accel_scale, body_field[1].acc[1] * accel_scale]), color = 0x4EA4EE)

    #velocity vectors
    gui.arrow(body_field[0].pos, ti.Vector([body_field[0].vel[0] * veloc_scale, body_field[0].vel[1] * veloc_scale]), color = 0xE05E26)
    gui.arrow(body_field[1].pos, ti.Vector([body_field[1].vel[0] * veloc_scale, body_field[1].vel[1] * veloc_scale]), color = 0xE05E26)

    gui.show()