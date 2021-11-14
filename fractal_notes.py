import taichi as ti

#https://docs.taichi.graphics/

#initializes the taichi context, allows it to find and use system GPU (will default to CPU if no GPU is found)
#the second argument allows taichi to use a given fraction of GPU memory (as opposed to the CUDA default of 1GB)
ti.init(arch=ti.gpu, device_memory_fraction=0.5)

#sets the pixel space to be a field of floats whose size is 2n by n (fields are just 2d arrays in taichi)
n = 320
pixels = ti.field(dtype=float, shape=(n*2, n))

@ti.func #this decorator tells the taichi compiler to optimize this function for the GPU
#taichi functions can ONLY be called by other taichi functions or kernels
#taichi functions can be nested but not recursive
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

@ti.kernel #this decorator tells the taichi compiler that this is not a function but a kernel (like an HLSL pragma kernel)
#kernel functions must be type hinted
#nested kernels are unsupported
def paint(t: float):
    for i, j in pixels: #parallelization across all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i/n - 1, j/n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

#setup gui details (name and x by y resolution)
gui = ti.GUI("Julia Set", res=(n*2, n))

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()