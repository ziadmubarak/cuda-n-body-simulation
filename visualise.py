import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

files = sorted(glob.glob("frames/frame_*.csv"))
print(f"Found {len(files)} frames")

frames = []
for f in files:
    data = np.loadtxt(f, delimiter=",")
    frames.append(data)

fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter([], [], s=2, c="white")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")

all_x = np.concatenate([f[:, 0] for f in frames])
all_y = np.concatenate([f[:, 1] for f in frames])
ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
ax.set_ylim(all_y.min() - 10, all_y.max() + 10)
ax.set_xticks([])
ax.set_yticks([])

def update(i):
    data = frames[i]
    scatter.set_offsets(data[:, :2])
    ax.set_title(f"Step {i*5}", color="white")
    return scatter,

ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
ani.save("nbody.gif", writer=PillowWriter(fps=20))
print("Saved nbody.gif")