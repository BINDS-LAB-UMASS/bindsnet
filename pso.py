import numpy as np
particle_positions = np.zeros([12,2])
particle_velocities = np.zeros([12,2])
for i in range(12):
    for dim in range(2):
        particle_positions[i][dim] = np.random.random_sample()*100
        particle_velocities[i][dim] = np.random.random_sample() * 200 - 100

best_positions = particle_positions
best_swarm_position = particle_positions[0]

np.savetxt("particle_pos.txt", particle_positions)
np.savetxt("particle_vel.txt", particle_velocities)
np.savetxt("particle_best.txt", best_positions)