from training import PINN 
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

y_min = -2.1
y_max = 2
x_min = -2
x_max = 2

pinn = PINN()

pinn.net.load_state_dict(torch.load(
        "c:/Users/DakotaBarnhardt/Downloads/Airfoils/Param.pt"))

step_size = .001

x = np.arange(x_min, x_max, step_size)
y = np.arange(y_min, y_max, step_size)

X, Y = np.meshgrid(x, y)

x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)

xyn = np.concatenate([x, y], axis=1)
xy = []
for x in xyn:
    if (((x[0] - 0.25)**2 + (x[1] - 0.04)**2)**(0.5)) < 0.02:
        xy.append([0, 0])
    else:
        xy.append([x[0], x[1]])

xy = np.array(xy)

xy = torch.tensor(xy, dtype=torch.float32).to(device)

with torch.no_grad():
    u, v, p = pinn.predict(xy)
    
    u = u.cpu().numpy()
    u = u.reshape(Y.shape)

    v = v.cpu().numpy()
    v = v.reshape(Y.shape)

    p = p.cpu().numpy()
    p = p.reshape(Y.shape)

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
data = (u, v, p)
labels = ["u(x,y)","v(x,y)", "p(x,y)"]
for i in range(3):
    ax = axes[i]
    im = ax.imshow(
        data[i], cmap="rainbow", 
        extent=[x_min, x_max, y_min, y_max], origin="lower"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="3%")
    fig.colorbar(im, cax=cax, label=labels[i])
    ax.set_title(labels[i])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

fig.tight_layout()

fig.savefig("c:/Users/DakotaBarnhardt/Downloads/Airfoils/Sol.png", dpi=500)



'''
import torch

# Example tensors representing x and y components
x = torch.tensor([3.0, 4.0])
y = torch.tensor([4.0, 3.0])

# Compute the magnitude (norm) of the vector
magnitude = torch.sqrt(x**2 + y**2)
print(magnitude)  # Output: tensor([5.0000, 5.0000])
'''