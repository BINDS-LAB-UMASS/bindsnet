import numpy as np
particle_positions = np.zeros([13,3])
particle_velocities = np.zeros([13,3])
best_positions = np.zeros([13,4])
for i in range(13):
    for dim in range(2):
        particle_positions[i][dim] = np.random.random_sample()*100
        particle_velocities[i][dim] = (np.random.random_sample() * 100 - particle_positions[i][dim])/2
        best_positions[i][dim] = particle_positions[i][dim]
    particle_positions[i][2] = np.random.random_sample() * 20
    particle_velocities[i][2] = (np.random.random_sample() * 20 - particle_positions[i][2])/2
    best_positions[i][2] = particle_positions[i][2]
    best_positions[i][3] = 0

np.savetxt("particle_pos.txt", particle_positions)
np.savetxt("particle_vel.txt", particle_velocities)
np.savetxt("particle_best.txt", best_positions)