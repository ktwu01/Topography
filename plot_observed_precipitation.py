import os
import matplotlib.pyplot as plt
import numpy as np

elev = np.array([20, 37, 230, 450, 550, 600, 640, 1150, 1500, 2350])
pr = np.array([129, 252, 421, 495, 290, 332, 680, 510, 510, 480])

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(pr,elev)
ax.set_xlim([0, 1000])
ax.set_ylim([0, 2200])
ax.set_xlabel("Pr [mm/yr]")
ax.set_ylabel("Elevation [m]")
plt.show()