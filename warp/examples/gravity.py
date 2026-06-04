import warp as wp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
args = parser.parse_args()

wp.set_device(args.device)

num_particles = 1_000_000
dt = 0.01

@wp.kernel
def gravity_step(pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    i = wp.tid()
    position = pos[i]
    dist_sq = wp.length_sq(position) + 0.01  # softened distance
    acc = -1000.0 / dist_sq * wp.normalize(position)  # gravitational pull toward origin
    vel[i] = vel[i] + acc * dt
    pos[i] = pos[i] + vel[i] * dt

rng = np.random.default_rng(42)
positions = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)
velocities = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)

for _ in range(100):
    wp.launch(gravity_step, dim=num_particles, inputs=[positions, velocities])

print(positions.numpy())