import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
from training import PINN 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from read_data import DAT

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

y_min = -.5
y_max = .5
x_min = -1
x_max = 2
airfoil_points= DAT.read_data('ah79100b.dat')

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
polygon = Polygon(airfoil_points)
for x in xyn:
    point = Point(x[0],x[1])
    if polygon.contains(point):
        xy.append([0, 0])
    else:
        xy.append([x[0], x[1]])

xy = np.array(xy)

xy = torch.tensor(xy, dtype=torch.float32).to(device)

with torch.no_grad():
    u, v, p,s1,s2,s3 = pinn.predict(xy)
    
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
        extent=[x_min, x_max, y_min, y_max],origin='lower'
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


