from math import pi
import taichi as ti

#init taichi context + window details
ti.init(arch=ti.gpu, device_memory_fraction=0.5)
n: int = 1200
pixels = ti.field(dtype=float, shape=(n, n)) #square screen

#simulation constants
g: float = 9.81 #gravitational constant
root = ti.Vector([0.5, 0.7])

#!!! TAICHI TRIG FUNCTIONS OPERATE ON RADIANS

nbodies: int = 2 #total pendulums in field
pendulum_field = ti.Struct.field({
    "length": ti.f32,
    "pos": ti.types.vector(2, ti.f32),
    "theta": ti.f32, #in radians
    "omega": ti.f32,
    "mass": ti.f32,
}, shape=(nbodies,))

#initialize independent state
pendulum_field[0].mass = 1
pendulum_field[1].mass = 1

pendulum_field[0].length = 1
pendulum_field[1].length = 0.7

pendulum_field[0].theta = pi/2
pendulum_field[1].theta = -pi/4

#initalize dependent state
pendulum_field[0].pos = root + ti.Vector([pendulum_field[0].length*ti.cos(pendulum_field[0].theta), pendulum_field[0].length*ti.sin(pendulum_field[0].theta)])
pendulum_field[1].pos = pendulum_field[0].pos + ti.Vector([pendulum_field[1].length*ti.cos(pendulum_field[1].theta), pendulum_field[0].length*ti.sin(pendulum_field[1].theta)])

@ti.func
def alpha_1():
    one = pendulum_field[0]
    two = pendulum_field[1]
    return (-g*(2*one.mass + two.mass)*ti.sin(one.theta)-two.mass*g*ti.sin(one.theta-2*two.theta)-2*ti.sin(one.theta-two.theta)*two.mass*(two.omega**2*two.length+one.omega**2*one.length*ti.cos(one.theta-two.theta)))/(one.length*(2*one.mass+two.mass-two.mass*ti.cos(2*one.theta-2*two.theta)))

@ti.func
def alpha_2():
    one = pendulum_field[0]
    two = pendulum_field[1]
    return (2*ti.sin(one.theta-two.theta)*(one.omega**2*one.length*(one.mass+two.mass)+g*(one.mass+two.mass)*ti.cos(one.theta)+two.omega**2*two.length*two.mass*ti.cos(one.theta-two.theta)))/(one.length*(2*one.mass+two.mass-two.mass*ti.cos(2*one.theta-2*two.theta)))

@ti.kernel
def update(dt: float):
    #using euler approximations to advance the simulation
    pendulum_field[0].omega += alpha_1() * dt
    pendulum_field[1].omega += alpha_2() * dt

    pendulum_field[0].theta += pendulum_field[0].omega * dt
    pendulum_field[1].theta += pendulum_field[1].omega * dt

gui = ti.GUI("Double Pendulum", res=(n, n))

for i in range(10000):
    gui.circle(root, radius = 7, color=0x068587)
    update(0.01)
    for i in range(nbodies):
        gui.circle(pendulum_field[i].pos, radius = 15, color=0x068587)
    gui.show()
