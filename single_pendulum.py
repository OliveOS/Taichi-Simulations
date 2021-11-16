import taichi as ti

#workings: https://www.desmos.com/calculator/0csyohcihd

#init taichi context + window details
ti.init(arch=ti.gpu, device_memory_fraction=0.5, excepthook=True)
n: int = 1200
pixels = ti.field(dtype=float, shape=(n, n))

#simulation constants
g: float = 9.81 #gravitational constant
pivot = ti.Vector([0.5, 0.5])

nbodies: int = 1 #total pendulums in field
pendulum_field = ti.Struct.field({
    "length": ti.f32,
    "pos": ti.types.vector(2, ti.f32),
    "vel": ti.types.vector(2, ti.f32),
    "acc": ti.types.vector(2, ti.f32),
    "mass": ti.f32,
}, shape=(nbodies,))

pendulum_field[0].length = 0.3
pendulum_field[0].pos = ti.Vector([0.8, 0.5])
pendulum_field[0].mass = 1


@ti.func
def get_equilibrium(r):
    return ti.Vector([pivot[0]-r, pivot[1]])

@ti.func
def get_theta(eq, pos, r):
    b = ti.sqrt((pos[0]-eq[0])**2 + (pos[1]-eq[1])**2)
    return ti.asin(b/r)

@ti.func
def update_pos_derivatives(dt):
    for i in range(pendulum_field):

        theta = get_theta(get_equilibrium(pendulum_field[i].length), pendulum_field[i].pos, pendulum_field[i].length)

        #set accelerations (not dependent on dt or previous values of acceleration)
        pendulum_field[i].acc[0] = g * ti.sin(theta) * ti.cos(theta)
        pendulum_field[i].acc[1] = g * ti.sin(theta)**2

        #set velocities
        pendulum_field[i].vel[0] = pendulum_field[i].acc[0] * dt
        pendulum_field[i].vel[1] = pendulum_field[i].acc[1] * dt

        #set positions
        pendulum_field[i].pos[0] = pendulum_field[i].vel[0] * dt
        pendulum_field[i].pos[1] = pendulum_field[i].vel[1] * dt

@ti.kernel 
def update(dt: float):
    update_pos_derivatives(dt)

#setup gui details (name and x by y resolution)
gui = ti.GUI("Pendulum", res=(n, n))

for i in range(10000):
    update(0.01)
    gui.show()