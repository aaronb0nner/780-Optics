#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift
from providedCode.make_long_otf import make_long_otf
from providedCode.make_otf2 import make_pupil, make_otf
from providedCode.turbulence import generate_zern_polys, make_screens, general_phase_screen
import matplotlib.pyplot as plt
#%%
# Generate 100 phase screens
frames = 100 # number of phase screens
nn = 300 # image size (pixels)
d = 7.0 # cm
ro = 1.0 # cm
l1 = d * 2 # size of coordinate grid (cm)
deltat = 1 # arbitray time between screens
windx = 6 # wind speed in x 
windy = 6 # wind speed in y
boil = 1 # atmospheric boil factor
zern_mx = 120

zern, ch = generate_zern_polys(zern_mx, nn, d, ro)

# Generate the phase screens from turbulence.py

phase_screens = np.zeros((frames, nn, nn))
for i in range(frames):
    phase_screens[i] = general_phase_screen(nn, zern_mx, zern, ch)

# Compute pixel scale
dx = d / nn # cm per pixel

# Create the pupil mask using the make_pupil function
pupil_radius_pixels = (d / 2) / dx
r2 = 0

pupil = make_pupil(pupil_radius_pixels, r2, nn)
pupil[nn // 2, nn // 2] = 1.0

# Compute OTF for each phase screen
otf_list = []
for i in range(frames):
    phase = phase_screens[i, :, :]
    cpupil = pupil * np.exp(1j * phase)
    otf, _ = make_otf(1, cpupil)
    otf_list.append(otf)

otf_array = np.array(otf_list) # shape: (frames, nn, nn)

# Average the Simulated OTFs
avg_otf = np.mean(otf_array, axis=0)

# Normalize the averaged OTF so the peak magnitude is 1
avg_otf /= np.max(np.abs(avg_otf))

# Generate Long Exposure OTFs over a Range of Seeing Parameters
ro_values = np.arange(0.5, 2.1, 0.1)  # cm (5mm to 2 cm in 1 mm increments)
long_otf_dict = {} # Create dictionary to store ro values and associated OTF

for ro_long in ro_values:
    ro_key = round(ro_long, 1)
    long_otf = make_long_otf(pupil_radius_pixels, dx, nn, ro_long)
    long_otf_dict[ro_key] = long_otf
                  
sse_list = [] # Store the SSE for each seeing parameter

for ro_long in ro_values:
    ro_key = round(ro_long, 1)
    long_otf = long_otf_dict[ro_key]
    sse = np.sum((np.abs(long_otf) - np.abs(avg_otf))**2)
    sse_list.append(sse)

# Plot the SSE vs. Seeing Parameter
plt.figure(figsize=(8, 6))
plt.plot(ro_values, sse_list, marker='o')
plt.xlabel('Seeing Parameter (ro) [cm]')
plt.ylabel('Sum Squared Error (SSE)')
plt.title('SSE between Long Exposure OTF and Average Simulated OTF vs. Seeing')
plt.grid(True)
plt.show()

# %%
