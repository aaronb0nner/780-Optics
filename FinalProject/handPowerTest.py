import numpy as np
import matplotlib.pyplot as plt

# Parameters
beam_radius = 0.03            # 3 cm Gaussian beam radius (1/e field), in meters
total_power = 2.5e-3          # Total power = 2.5 mW
sampling = 1.55e-6            # Sampling rate in meters
sensor_pixels = 2000          # Sensor is 2000 x 2000 pixels
sensor_size = sensor_pixels * sampling  # Size of sensor in meters

# Grid: 6 cm diameter, so 3 cm radius
grid_radius = 0.03
grid_size = int((2 * grid_radius) / sampling)
if grid_size % 2 == 1:
    grid_size += 1  # ensure even number of points for symmetric center

# Create coordinate grid centered at (0,0)
x = np.linspace(-grid_radius, grid_radius, grid_size)
y = np.linspace(-grid_radius, grid_radius, grid_size)
X, Y = np.meshgrid(x, y)

# Unnormalized 2D Gaussian profile
I = np.exp(-2 * (X**2 + Y**2) / beam_radius**2)

# Normalize to 2.5 mW total power
pixel_area = sampling**2
I_sum = np.sum(I * pixel_area)
I_normalized = I * (total_power / I_sum)

# Extract central sensor region
start = grid_size // 2 - sensor_pixels // 2
end = start + sensor_pixels
sensor_power = np.sum(I_normalized[start:end, start:end] * pixel_area)

# Output power seen by sensor
print(f"Total beam power: {total_power * 1e3:.3f} mW")
print(f"Power captured by sensor: {sensor_power * 1e3:.3f} mW")
print(f"Percentage captured: {(sensor_power / total_power) * 100:.2f}%")

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
extent = [-grid_radius, grid_radius, -grid_radius, grid_radius]
im = ax.imshow(I_normalized, cmap='gray', extent=extent, origin='lower')
rect_half = sensor_size / 2
rect = plt.Rectangle((-rect_half, -rect_half), sensor_size, sensor_size,
                     linewidth=1.5, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title('2D Gaussian Beam Profile with Sensor Overlay')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
plt.colorbar(im, ax=ax, label='Irradiance (W/m²)')
plt.tight_layout()
plt.show()
