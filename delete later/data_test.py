import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import torch
import matplotlib.patches as patches



data2 = [[0,0],
[0.00107,0.00544],
[	0.00428	,	0.01064	],
[	0.00961	,	0.01683	],
[	0.01704	,	0.02346	],
[	0.02653	,	0.03048	],
[	0.03806	,	0.03741	],
[	0.05156	,	0.04463	],
[	0.06699	,	0.05204	],
[	0.08427	,	0.05944	],
[	0.10332	,	0.06652	],
[	0.12408	,	0.07342	],
[	0.14645	,	0.07979	],
[	0.17033	,	0.08583	],
[	0.19562	,	0.09121	],
[	0.22221	,	0.09614	],
[	0.25	,	0.10031	],
[	0.27886	,	0.10386	],
[	0.30866	,	0.10652	],
[	0.33928	,	0.10852	],
[	0.37059	,	0.10964	],
[	0.40245	,	0.11015	],
[	0.43474	,	0.1098	],
[	0.4673	,	0.10879	],
[	0.5	,	0.10692	],
[	0.5327	,	0.1044	],
[	0.56526	,	0.10105	],
[	0.59755	,	0.09709	],
[	0.62941	,	0.09238	],
[	0.66072	,	0.08721	],
[	0.69134	,	0.08155	],
[	0.72114	,	0.07565	],
[	0.75	,	0.06952	],
[	0.77779	,	0.06344	],
[	0.8043801	,	0.05733	],
[	0.82967	,	0.05138	],
[	0.85355	,	0.04548	],
[	0.8759201	,	0.03985	],
[	0.89668	,	0.03437	],
[	0.91573	,	0.02923	],
[	0.93301	,	0.02428	],
[	0.94844	,	0.01968	],
[	0.96194	,	0.01529	],
[	0.97347	,	0.01129	],
[	0.98296	,	0.00763	],
[	0.99039	,	0.00444	],
[	0.99572	,	0.00196	],
[	0.99893	,	0.00031	],
[	1	,	0	],
[	0	,	0	],
[	0.00107	,	-0.00324	],
[	0.00428	,	-0.00624	],
[	0.00961	,	-0.00874	],
[	0.01704	,	-0.01015	],
[	0.02653	,	-0.01079	],
[	0.03806	,	-0.01087	],
[	0.05156	,	-0.01059	],
[	0.06699	,	-0.00995	],
[	0.08427	,	-0.00904	],
[	0.10332	,	-0.00793	],
[	0.12408	,	-0.00668	],
[	0.14645	,	-0.00524	],
[	0.17033	,	-0.0037	],
[	0.19562	,	-0.00198	],
[	0.22221	,	-0.00015	],
[	0.25	,	0.00185	],
[	0.27886	,	0.00391	],
[	0.30866	,	0.00615	],
[	0.33928	,	0.00847	],
[	0.37059	,	0.01099	],
[	0.40245	,	0.01356	],
[	0.43474	,	0.01622	],
[	0.4673	,	0.01885	],
[	0.5	,	0.02147	],
[	0.5327	,	0.02391	],
[	0.56526	,	0.0261	],
[	0.59755	,	0.0279	],
[	0.62941	,	0.02936	],
[	0.66072	,	0.03037	],
[	0.69134	,	0.03099	],
[	0.72114	,	0.03117	],
[	0.75	,	0.03097	],
[	0.77779	,	0.03036	],
[	0.8043801	,	0.02937	],
[	0.82967	,	0.02797	],
[	0.85355	,	0.02619	],
[	0.8759201	,	0.02407	],
[	0.89668	,	0.02169	],
[	0.91573	,	0.01911	],
[	0.93301	,	0.01639	],
[	0.94844	,	0.01357	],
[	0.96194	,	0.01071	],
[	0.97347	,	0.0079	],
[	0.98296	,	0.00527	],
[	0.99039	,	0.0031	],
[	0.99572	,	0.00136	],
[	0.99893	,	0.00016	],
[	1	,	0	]]

def normal_data(data):

    previous=None
    next=data[2]
    normals=[]
    

    for i in range(len(data)):
        pair=data[i] 
        
        plt.plot(pair[0],pair[1],'o')
        if previous != None:
            if next!= None:
                normalx=next[0]-previous[0]
                normaly=next[1]-previous[1]
                plt.quiver(pair[0],pair[1],normaly,-normalx)
                normals.append([normaly,-normalx])
            else:
                next=data[1]
                normalx=next[0]-previous[0]
                normaly=next[1]-previous[1]
                plt.quiver(pair[0],pair[1],0,1)
                normals.append([0,1])
        
        previous=pair 
        if i+2<len(data):
            next=data[i+2]
        else:
            next=None
    return normals
    

def read_data(path):
    with open(path,'r') as f:
        fileLines = f.readlines()

    fileLines.pop(0)

    airfoilPoints=[]
    for line in fileLines:
        point=line.split(' ')
        while point.__contains__(''):
            point.remove('')
        point[0]=float(point[0])
        point[1]=float(point[1][0:len(point[1])-2])
        airfoilPoints.append(point)
    return airfoilPoints

normal_data(read_data('ah79100b.dat'))
plt.show()

u=torch.tensor([[1],[0],[3],[4],[5]])
v=torch.tensor([[0],[1],[7],[6],[4]])
uv=torch.concatenate([u,v], axis=1)
n_data=torch.tensor([[0,1],[-1,0],[-7,3],[-6,4],[-4,5]])
x=uv*n_data

x=torch.sum(x,dim=1)



fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

polygon_vertices = read_data('ah79100b.dat')

# Create a Polygon patch
polygon = patches.Polygon(polygon_vertices, closed=True, fill=True, edgecolor='r', facecolor='g', alpha=0.5)

# Add the polygon to the first subplot
plt.add_patch(polygon)

# Set limits for the axes
plt.set_xlim(0, 1)
plt.set_ylim(-.1, 1)

# Display the plot
plt.show()


x_min=0
x_max=10
y_min=0
y_max=10
step_size=1
x = np.arange(x_min, x_max, step_size)
y = np.arange(y_min, y_max, step_size)
print('1 x',x,'\n1y',y)
X, Y = np.meshgrid(x, y)
print('\n2 X',X,'\n 2 Y',Y)

x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)
print('\n3 x',x,'\n3y',y)

xyn = np.concatenate([x, y], axis=1)
print('4xyn',xyn)

xyn = np.array(xyn)
print('5xyn',xyn)
xy = torch.tensor(xyn, dtype=torch.float32)
print('6xy',xy)