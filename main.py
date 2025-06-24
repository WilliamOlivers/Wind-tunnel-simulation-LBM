import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
height, width = 80, 200
viscosity = 0.02
omega = 1 / (3 * viscosity + 0.5)
u0 = 0.1
four9ths, one9th, one36th = 4/9, 1/9, 1/36

# Create cylinder barrier
def create_cylinder():
    barrier = np.zeros((height, width), dtype=bool)
    center_x, center_y, radius = width//4, height//2, 8
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    barrier[mask] = True
    return barrier

def create_rectangle():
    barrier = np.zeros((height, width), dtype=bool)
    center_x, center_y = width//4, height//2
    length_x, length_y = 16, 12  # width and height of the rectangle
    
    y, x = np.ogrid[:height, :width]
    
    # Conditions to be in the rectangle
    mask = (abs(x - center_x) <= length_x//2) & (abs(y - center_y) <= length_y//2)
    
    barrier[mask] = True
    return barrier

# Initial conditions
def equilibrium(w, u):
    return w * (1 + 3*u + 4.5*u**2 - 1.5*u0**2)

def initialize():
    global n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW, rho, ux, uy
    global barrier, barrierN, barrierS, barrierE, barrierW, barrierNE, barrierNW, barrierSE, barrierSW
    
    barrier = create_cylinder()
    
    ux_init = u0 * np.ones((height, width))
    ux_init_neg = -u0 * np.ones((height, width))

    n0  = four9ths * (np.ones((height, width)) - 1.5 * u0**2)
    nN  = one9th   * (np.ones((height, width)) - 1.5 * u0**2)
    nS  = one9th   * (np.ones((height, width)) - 1.5 * u0**2)
    nE  = equilibrium(one9th,  ux_init)
    nW  = equilibrium(one9th, ux_init_neg)
    nNE = equilibrium(one36th, ux_init)
    nSE = equilibrium(one36th, ux_init)
    nNW = equilibrium(one36th, ux_init_neg)
    nSW = equilibrium(one36th, ux_init_neg)

    rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW
    ux  = (nE + nNE + nSE - nW - nNW - nSW) / rho
    uy  = (nN + nNE + nNW - nS - nSE - nSW) / rho

    def shifted(barrier, dx, dy):
        return np.roll(np.roll(barrier, dy, axis=0), dx, axis=1)

    barrierN  = shifted(barrier, 0,  1)
    barrierS  = shifted(barrier, 0, -1)
    barrierE  = shifted(barrier, 1,  0)
    barrierW  = shifted(barrier,-1,  0)
    barrierNE = shifted(barrierN,  1, 0)
    barrierNW = shifted(barrierN, -1, 0)
    barrierSE = shifted(barrierS,  1, 0)
    barrierSW = shifted(barrierS, -1, 0)

def stream():
    global nN, nS, nE, nW, nNE, nNW, nSE, nSW

    def roll(n, dx, dy):
        return np.roll(np.roll(n, dy, axis=0), dx, axis=1)

    nN  = roll(nN,  0,  1)
    nS  = roll(nS,  0, -1)
    nE  = roll(nE,  1,  0)
    nW  = roll(nW, -1,  0)
    nNE = roll(nNE, 1,  1)
    nNW = roll(nNW,-1,  1)
    nSE = roll(nSE, 1, -1)
    nSW = roll(nSW,-1, -1)

    nN[barrierN]  = nS[barrier]
    nS[barrierS]  = nN[barrier]
    nE[barrierE]  = nW[barrier]
    nW[barrierW]  = nE[barrier]
    nNE[barrierNE] = nSW[barrier]
    nNW[barrierNW] = nSE[barrier]
    nSE[barrierSE] = nNW[barrier]
    nSW[barrierSW] = nNE[barrier]

def collide():
    global rho, ux, uy, n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW

    rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW
    ux = (nE + nNE + nSE - nW - nNW - nSW) / rho
    uy = (nN + nNE + nNW - nS - nSE - nSW) / rho

    ux2, uy2 = ux**2, uy**2
    u2 = ux2 + uy2
    omu215 = 1 - 1.5 * u2
    uxuy = ux * uy

    n0  = (1 - omega) * n0  + omega * four9ths * rho * omu215
    nN  = (1 - omega) * nN  + omega * one9th * rho * (omu215 + 3*uy + 4.5*uy2)
    nS  = (1 - omega) * nS  + omega * one9th * rho * (omu215 - 3*uy + 4.5*uy2)
    nE  = (1 - omega) * nE  + omega * one9th * rho * (omu215 + 3*ux + 4.5*ux2)
    nW  = (1 - omega) * nW  + omega * one9th * rho * (omu215 - 3*ux + 4.5*ux2)
    nNE = (1 - omega) * nNE + omega * one36th * rho * (omu215 + 3*(ux+uy) + 4.5*(u2 + 2*uxuy))
    nNW = (1 - omega) * nNW + omega * one36th * rho * (omu215 + 3*(-ux+uy) + 4.5*(u2 - 2*uxuy))
    nSE = (1 - omega) * nSE + omega * one36th * rho * (omu215 + 3*(ux-uy) + 4.5*(u2 - 2*uxuy))
    nSW = (1 - omega) * nSW + omega * one36th * rho * (omu215 + 3*(-ux-uy) + 4.5*(u2 + 2*uxuy))

    ux_boundary_pos = u0 * np.ones(height)
    ux_boundary_neg = -u0 * np.ones(height)
    
    nE[:,0]  = equilibrium(one9th,   ux_boundary_pos)
    nW[:,0]  = equilibrium(one9th,  ux_boundary_neg)
    nNE[:,0] = equilibrium(one36th,  ux_boundary_pos)
    nSE[:,0] = equilibrium(one36th,  ux_boundary_pos)
    nNW[:,0] = equilibrium(one36th, ux_boundary_neg)
    nSW[:,0] = equilibrium(one36th, ux_boundary_neg)

# Initialize
initialize()

# Setup plot
fig, ax = plt.subplots(figsize=(12, 6))
curl = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1) - np.roll(ux, -1, axis=0) + np.roll(ux, 1, axis=0)
fluidImage = ax.imshow(curl, origin='lower', cmap='hot', norm=plt.Normalize(-0.1, 0.1))

barrierRGBA = np.zeros((height, width, 4), dtype=np.uint8)
barrierRGBA[barrier, :3] = [0, 0, 0]
barrierRGBA[barrier, 3] = 255
barrierImage = ax.imshow(barrierRGBA, origin='lower')

ax.set_xlim(0, width)
ax.set_ylim(0, height)

def animate(frame):
    for _ in range(20):  # 20 times per seconds
        stream()
        collide()
    
    curl = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1) - np.roll(ux, -1, axis=0) + np.roll(ux, 1, axis=0)
    fluidImage.set_array(curl)
    
    return [fluidImage, barrierImage]

anim = animation.FuncAnimation(fig, animate, interval=50, blit=False)
plt.show()