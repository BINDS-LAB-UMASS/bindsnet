import numpy as np
particle_positions = np.zeros([12,2])
particle_velocities = np.zeros([12,2])
best_positions = np.zeros([12,3])
for i in range(12):
    for dim in range(2):
        particle_positions[i][dim] = np.random.random_sample()*100
        particle_velocities[i][dim] = (np.random.random_sample() * 100 - particle_positions[i][dim])/2
        best_positions[i][dim] = particle_positions[i][dim]
    best_positions[i][2] = 0

np.savetxt("particle_pos.txt", particle_positions)
np.savetxt("particle_vel.txt", particle_velocities)
np.savetxt("particle_best.txt", best_positions)