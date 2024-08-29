import vtuIO
'''
# Load the VTU file
vtufile = vtuIO.VTUIO("flow.vtu", dim=3)

# Access data
points = vtufile.GetPoints()
cells = vtufile.GetCells()
'''
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from mpl_toolkits.axes_grid1 import make_axes_locatable



# Read the VTU file
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("flow.vtu")
reader.Update()

# Get the coordinates of nodes in the mesh
vtk_array= reader.GetOutput().GetPoints().GetData()

#The "Temperature" field is the third scalar in my vtk file
pressure_vtk_array = reader.GetOutput().GetPointData().GetArray(0)
velocity_vtk_array = reader.GetOutput().GetPointData().GetArray(1)
pressure_coefficient_vtk_array = reader.GetOutput().GetPointData().GetArray(2)
density_vtk_array = reader.GetOutput().GetPointData().GetArray(3)

numpy_array = vtk_to_numpy(vtk_array)
pressure_numpy_array = vtk_to_numpy(pressure_vtk_array)
velocity_numpy_array = vtk_to_numpy(velocity_vtk_array)
print(velocity_numpy_array[0:5])
pressure_coefficient_numpy_array = vtk_to_numpy(pressure_coefficient_vtk_array)
density_numpy_array = vtk_to_numpy(density_vtk_array)
count=0
ox=[sublist[0] for sublist in numpy_array]
oy=[sublist[1] for sublist in numpy_array]
ou=[sublist[0] for sublist in velocity_numpy_array]
ov=[sublist[1] for sublist in velocity_numpy_array]
print(ou[0:5],ov[0:5])
op=pressure_numpy_array
nx=[]
ny=[]
nu=[]
nv=[]
npr=[]
for x,y,u,v,p in zip(ox,oy,ou,ov,op):
    if -.2<x<1 and -.05<y<.15:
        nx.append(x)
        ny.append(y)
        nu.append(u)
        nv.append(v)
        npr.append(p)
        count+=1
X, Y = np.meshgrid(nx, ny)
colors = [npr,nu,nv]

fig, axes = plt.subplots(3, 1)
labels = ["p(x,y)","u(x,y)","v(x,y)"]
for i in range(3):
    ax = axes[i]
    #print(X.shape,Y.shape,len(colors[i]))
    #ax.pcolormesh(X,Y,colors[i])
    #ax.show()
    im=ax.scatter(nx, ny, c=colors[i], cmap='rainbow')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="3%")
    fig.colorbar(im, cax=cax, label=labels[i])
    ax.set_title(labels[i])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(-.2,1)
    ax.set_ylim(-.05,.15)
plt.show()


