import taichi as ti

#init taichi context + window details
ti.init(arch=ti.gpu, device_memory_fraction=0.5)
n: int = 1200
pixels = ti.field(dtype=float, shape=(n, n)) #square screen

#simulation constants
g: float = 9.81 #gravitational constant
root: ti.Vector([n/2, 0.75*n])

nbodies: int = 2 #total pendulums in field
pendulum_field = ti.Struct.field({
    "length": ti.f32,
    "pos": ti.types.vector(2, dt=ti.f32),
    "theta": ti.f32, #in degrees
    "omega": ti.f32,
    "mass": ti.f32,
}, shape=(nbodies,))

#set initial independent state
pendulum_field[0].length = 1
pendulum_field[1].length = 0.7
pendulum_field[0].theta = -90
pendulum_field[1].theta = 20
pendulum_field[0].omega = 0
pendulum_field[1].omega = 0
pendulum_field[0].mass = 1
pendulum_field[1].mass = 1

def polar_to_cartesian(r, theta):
    return ti.Vector([r*ti.cos(theta), r*ti.sin(theta)])

#set initial position of pendula

#TODO:Write update-omega function
#TODO:Write update-theta function
#TODO:Resolve functions into kernel-scope