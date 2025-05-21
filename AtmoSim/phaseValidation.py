import numpy as np
import matplotlib.pyplot as plt
from providedCode.turbulence import general_phase_screen, generate_zern_polys
from providedCode.make_otf2 import make_otf, make_pupil
from providedCode.make_long_otf import make_long_otf

# Parameters
aperture_diameter = 0.07  # meters (7 cm)
seeing_range = np.arange(0.005, 0.021, 0.001)  # meters (5 mm to 20 mm)
num_screens = 200
si = 300  # grid size (pixels)
dx = aperture_diameter / si  # meters per pixel

# Phase screen parameters
ro = 0.01  # meters (10 mm)
zern_mx = 150  # increased for better fidelity
l1 = aperture_diameter * 2
deltat = 1
windx = 6
windy = 6
boil = 1

# Generate Zernike polynomials and Cholesky decomposition
zern, ch = generate_zern_polys(zern_mx, si, aperture_diameter, ro)

# Generate phase screens
phase_screens = np.zeros((num_screens, si, si))
for i in range(num_screens):
    phase_screens[i] = general_phase_screen(si, zern_mx, zern, ch)

# Create pupil mask
pupil_radius_pixels = (aperture_diameter / 2) / dx
pupil = make_pupil(pupil_radius_pixels, 0, si)
pupil[si // 2, si // 2] = 1.0

# Generate OTFs for each phase screen
otfs = []
for i in range(num_screens):
    cpupil = pupil * np.exp(1j * phase_screens[i])
    otf, _ = make_otf(1, cpupil)
    otfs.append(otf)
otfs = np.array(otfs)

# Average and normalize
avg_otf_zernike = np.mean(otfs, axis=0)
avg_otf_zernike_norm = avg_otf_zernike / np.max(np.abs(avg_otf_zernike))
#avg_otf_zernike_norm = np.abs(avg_otf_zernike)
# Compare to long-exposure OTFs
sse_list = []
for ro_test in seeing_range:
    long_otf = make_long_otf(aperture_diameter, dx, si, ro_test)
    sse = np.sum((np.abs(long_otf) - np.abs(avg_otf_zernike_norm)) ** 2)
    sse_list.append(sse)

# Plot SSE curve
plt.figure(figsize=(8, 6))
plt.plot(seeing_range * 1000, sse_list, marker='o')
plt.xlabel('Seeing (mm)')
plt.ylabel('Sum Squared Error')
plt.title('SSE between Zernike OTF and Long OTF vs. Various Seeing Parameters')
plt.grid(True)
plt.show()
