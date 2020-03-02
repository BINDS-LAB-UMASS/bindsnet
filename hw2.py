import numpy as np
from matplotlib import pyplot as plt  
from matplotlib import style 

r = 10
w = 6
d = 1
num_samples = 2000

n_samples_out = n_samples // 2
n_samples_in = n_samples - n_samples_out

outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) + d
inner_circ_x = r - np.cos(np.linspace(0, np.pi, n_samples_in))
inner_circ_y = 0.5 - np.sin(np.linspace(0, np.pi, n_samples_in)) - d

X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

plt.scatter(X[:, 0], X[:, 1], s = 40, color ='g') 
plt.xlabel("X") 
plt.ylabel("Y") 
  
plt.show() 
plt.clf() 