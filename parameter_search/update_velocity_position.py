import numpy as np

pos = np.loadtxt("particle_pos.txt")
vel = np.loadtxt("particle_vel.txt")
best = np.loadtxt("particle_best.txt")
omega = 1/(2 * np.log(2))
c = 1/2 + np.log(2)

g_pos = pos[0]
g = 0

for index, x in enumerate(pos):
    particle = np.loadtxt("analysis/snn_tgd_param_prob_"+str(x[0])+"x"+str(x[1])+"x.txt")
    perf = np.mean(particle)
    if best[index][2] < np.mean(perf):
        best[index] = [x[0], x[1], perf]
    if g < best[index][2]:
        g = best[index][2]
        g_pos = [best[index][0], best[index][1]]

new_vel = np.zeros_like(vel)
new_pos = np.zeros_like(pos)
for index, x in enumerate(pos):
    for dim in range(2):
        new_vel[index][dim] = omega * vel[index][dim] + (np.random.random_sample() * c  * (best[index][dim] - x[dim])) + (np.random.random_sample() * c  * (g_pos[dim] - x[dim]))
        new_pos[index][dim] = x[dim] + new_vel[index][dim]
        if new_pos[index][dim] < 0:
            new_pos[index][dim] = 0.1

np.savetxt("particle_pos.txt", new_pos)
np.savetxt("particle_vel.txt", new_vel)
np.savetxt("particle_best.txt", best)
