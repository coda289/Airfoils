from training import PINN 
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#reads the airfoil data 
# given as a path to a .dat file
def read_data(path):

    with open(path,'r') as f:
        file_lines = f.readlines()
    file_lines.pop(0)
    airfoil_points=[]

    for line in file_lines:
        point = line.split(' ')
        while point.__contains__(''):
            point.remove('')
        point[0] = float(point[0])
        point[1] = float(point[1][0:len(point[1])-2])
        airfoil_points.append(point)
    return airfoil_points

y_min = -1
y_max = 1
x_min = -1
x_max = 2
airfoil_points= read_data('ah79100b.dat')

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
polygon = patches.Polygon(airfoil_points, closed=True, fill=True, edgecolor='w', facecolor='w', alpha=0.5)
for i in range(3):
    ax = axes[i]
    #polygon = patches.Polygon(airfoil_points, closed=True, fill=True, edgecolor='w', facecolor='w', alpha=0.5)
    #ax.add_patch(polygon)
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