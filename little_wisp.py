import taichi as ti

#init taichi context + window details
ti.init(arch=ti.gpu, device_memory_fraction=0.8, excepthook=True)
n: int = 1200
pixels = ti.Vector.field(3, shape=(n, n), dtype = float) #n by n field of RGB values
diffusion_map = ti.field(dtype=float, shape=(n,n))
diffusion_map.fill(0.0)

#simulation constants
diff_color = ti.Vector([0.7, 0.7, 0.7])
diff_const = 0.3
brush_radius = 20

@ti.func #euclidean vector distance
def vector2d_distance(vec1, vec2):
    return ti.sqrt((vec2[1]-vec1[1])**2 + (vec2[0]-vec1[0])**2)

@ti.func #circular fill pattern, acts on the diffusion map
def circle_fill(center, radius):
    for i, j in pixels:
        if vector2d_distance(center, ti.Vector([i, j])) <= radius:
            diffusion_map[i, j] = 1.0

@ti.func
def diffuse_cell(i, j):
    if i-1 >= 0 and i-1 <= n:
        diffusion_map[i-1, j] += diff_const * diffusion_map[i, j]
    if i+1 >= 0 and i+1 <= n:
        diffusion_map[i+1, j] += diff_const * diffusion_map[i, j]
    if j-1 >= 0 and j-1 <= n:
        diffusion_map[i, j-1] += diff_const * diffusion_map[i, j]
    if j+1 >= 0 and j+1 <= n:
        diffusion_map[i, j+1] += diff_const * diffusion_map[i, j]

@ti.kernel 
def paint(cx: float, cy: float):
    circle_fill(ti.Vector([cx, cy]), brush_radius)
    for i, j in pixels:
        diffuse_cell(i, j)
        pixels[i, j] = diffusion_map[i, j] * diff_color

gui = ti.GUI("Wisp", res = (n, n))

for i in range(100000):
    gui.get_events()
    mpos = ti.Vector(gui.get_cursor_pos())
    paint(mpos[0]*n, mpos[1]*n)
    gui.set_image(pixels)
    gui.show()