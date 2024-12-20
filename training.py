import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from read_data import CSV, DAT, VTU
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from torch.autograd import grad

#set seed and device 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1234)
np.random.seed(1234)

#The min and max values of x and y
y_min = -.6
y_max = .6
x_min = -1.1
x_max = 2.1

#constants
Re = 2000
mu = 0.0000181206
rho = 1.22500

#Amount of points taken along boundry condition
bound_num = 100
#Amount of points to evaluate for PDE
points_num = 3000

#path DAT and CSV files
path_to_points = 'ah79100b.dat'
path_to_actual = 'flow.csv'

#reads and returns tensors for CVS
if('csv' in path_to_actual):
    validation_data = CSV.read_data(path_to_actual)
    uvp_list, xy_actual = CSV.tsplit_data(validation_data)
    uvp_act = uvp_list[2:5]
elif('vtu' in path_to_actual):
    xy_actual,uvp_act = VTU.read_data(path_to_actual)

#turns the data into an array and plots
airfoil_points = DAT.read_data(path_to_points)
for p in airfoil_points:
    plt.scatter(p[0],p[1],marker='o')

#turns the array into a tensor
t_airfoil_points = torch.tensor(
    airfoil_points,dtype=torch.float32,requires_grad=True)

def make_graph(data):
    #returns points for boundry
    #return random points outside the airfoil

    #walls
    front_wally = torch.linspace(y_min, y_max, bound_num)
    front_wallx = torch.ones_like(front_wally)*x_min
    top_wallx = torch.linspace(x_min, x_max, bound_num)
    top_wally = torch.ones_like(front_wally)*y_max
    bottom_wallx = torch.linspace(x_min, x_max, bound_num)
    bottom_wally = torch.ones_like(front_wally)*y_min

    #graph points
    plt.scatter(front_wallx, front_wally, marker='o')
    plt.scatter(top_wallx, top_wally, marker='o')
    plt.scatter(bottom_wallx, bottom_wally, marker='o')

    #boundary conditions
    inlet = torch.stack((front_wallx, front_wally), dim=1)
    c2 = torch.stack((top_wallx, top_wally), dim=1)
    c3 = torch.stack((bottom_wallx, bottom_wally), dim=1)
    #if top and bottom u=uinf
    #bc1 = torch.concatenate([inlet,c2,c3])
    #if top and bottom walls =0
    bc1 = torch.concatenate([c2,c3])
    bc1.clone().detach().requires_grad_(True)

    out_y = torch.linspace(y_min, y_max, bound_num)
    out_x = torch.ones_like(front_wally)*x_max
    plt.scatter(out_x, out_y, marker='o')
    outlet=torch.stack((out_x, out_y), dim=1)

    #make random 
    x = []
    y = []
    xy = []
    polygon = Polygon(data)
    #generate and validate data
    while len(xy)<points_num:
        xp = np.random.uniform(x_min, x_max)
        yp = np.random.uniform(y_min, y_max)
        point = Point(xp,yp)
        if not polygon.contains(point):
            x.append(xp)
            y.append(yp)
            xy.append([xp, yp])
    
    rand_points = torch.tensor(xy,dtype=torch.float32)
    plt.scatter(x,y,marker='o',c='red')

    #return tensors
    return bc1,rand_points,outlet,inlet

#generated points
bc1,rand_points,outlet,inlet = make_graph(airfoil_points)

def plotLoss(losses_dict, path, info=["I.C.", "B.C.", "OUT","P.D.E."]):
    #plots the loss
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(4), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)


def weights_init(m):
    #initalizes the weights as random
    #initalize bias as 0
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class layer(nn.Module):
    #layer class

    def __init__(self, n_in, n_out, activation):
        #layer initialize
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation
    
    def forward(self, x):
        #forward pass
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x


class DNN(nn.Module):
    #Model class
    
    def __init__(self, dim_in, dim_out, n_layer, n_node, activation=nn.Tanh()):
        #initilize
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.net.apply(weights_init)

    def forward(self, x):
        #forward pass
        out = x
        for layer in self.net:
            out = layer(out)
        return out


class PINN:
    #physics model 

    #constants 
    U_inf =1# Re*mu/rho
    AoA = 0
    
    def __init__(self) -> None:
        #intialize NN, optimizer, loss, and iteration
        self.net = DNN(dim_in=2, dim_out=6, n_layer=10, n_node=128).to(
            device
        )

        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=10000,
            max_eval=10000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc1": [], "bc2": [], "out": [],"pde":[]}
        self.iter = 0

    def predict(self, X):
        #predicts the output

        out = self.net(X)

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        
        sig_xx = out[:, 3:4]
        sig_xy = out[:, 4:5]
        sig_yy = out[:, 5:6]
    
        return u, v, p, sig_xx, sig_xy, sig_yy
    
    def out_loss(self,X):
        #outlet loss
        #presure = 0 at the right wall

        p = self.predict(X)[2]

        mse_outlet = torch.mean(torch.square(p))

        return mse_outlet

    def bc_loss1(self, X,U_inf=U_inf,AoA=AoA):
        #inlet loss at left wall
        #u=u_inf v=0 
        u, v = self.predict(X)[0:2]
        
        mse_bc = torch.mean(torch.square(u - U_inf*np.cos(AoA))) + torch.mean(
            torch.square(v - U_inf*np.sin(AoA))
        )

        return mse_bc
    
    def bc_loss2(self,X):
        #loss at top and bottom wall and airfoil
        #(u,v)=(0,0)
        u,v = self.predict(X)[0:2]
        
        mse_bc = torch.mean(torch.square(u) + torch.square(v))

        return mse_bc
    
    def pde_loss(self, X):
        #physics loss based on Navier stokes
        X = X.clone()
        X.requires_grad = True

        u, v, p, sig_11, sig_12, sig_22 = self.predict(X)

        u_out = grad(u.sum(), X, create_graph=True)[0]
        v_out = grad(v.sum(), X, create_graph=True)[0]

        sig_11_out = grad(sig_11.sum(), X, create_graph=True)[0]
        sig_12_out = grad(sig_12.sum(), X, create_graph=True)[0]
        sig_22_out = grad(sig_22.sum(), X, create_graph=True)[0]

        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]
        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]

        sig_11_x = sig_11_out[:, 0:1]
        sig_12_x = sig_12_out[:, 0:1]
        sig_12_y = sig_12_out[:, 1:2]
        sig_22_y = sig_22_out[:, 1:2]

        f0 = u_x + v_y

        f1 = (u*u_x + v*u_y) - sig_11_x - sig_12_y
        f2 = (u*v_x + v*v_y) - sig_12_x - sig_22_y

        f3 = -p + (2/Re) * u_x - sig_11
        f4 = -p + (2/Re) * v_y - sig_22
        f5 = (1/Re) * (u_y + v_x) - sig_12

        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_f3 = torch.mean(torch.square(f3))
        mse_f4 = torch.mean(torch.square(f4))
        mse_f5 = torch.mean(torch.square(f5))

        mse_pde = mse_f0 + mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5

        return mse_pde
    
    def mse_loss(self, X, actual):
        #mean squared error
        #predicted-actual

        X = X.clone()
        #pred_u, pred_v, pred_p,s1,s2,s3 = self.predict(X)
        pred_u, pred_v, pred_p = self.predict(X)
        
        loss_func = nn.MSELoss()
        u,v,p = actual[0], actual[1], actual[2]
        u = u.view(-1,1)
        v = v.view(-1,1)
        p = p.view(-1,1)

        loss1 = loss_func(pred_u,u)
        loss2 = loss_func(pred_v,v)
        loss3 = loss_func(pred_p,p)    

        loss=loss1+loss2+loss3
        
        print('u loss',torch.mean(loss1))
        print('v loss',torch.mean(loss2))
        print('p loss',torch.mean(loss3))
        print('total',loss)
        
        return loss,loss1,loss2,loss3

    def closure(self):
        #to run in the training loop

        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        #if top and bottom walls u=u_inf
        #mse_bc1 = self.bc_loss1(bc1)
        #mse_bc2 = self.bc_loss2(t_airfoil_points)
        #if top and bottom walls =0
        mse_bc1 = self.bc_loss1(inlet)
        mse_bc2 = self.bc_loss2(torch.cat([bc1,t_airfoil_points]))
        mse_pde = self.pde_loss(rand_points)
        mse_out = self.out_loss(outlet)

        loss = mse_bc1*10 + mse_bc2*10 + mse_pde+ mse_out

        loss.backward()

        #save loss
        self.losses["bc1"].append(mse_bc1.detach().cpu().item())
        self.losses["bc2"].append(mse_bc2.detach().cpu().item())
        self.losses["out"].append(mse_out.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        
        self.iter +=  1

        if(self.iter%200 == 0):
            print(f" It: {self.iter} Loss: {loss.item():.5e} BC1: {mse_bc1.item():.3e} BC2: {mse_bc2.item():.3e} OUT:{mse_out.item():.3e} PDE: {mse_pde.item():.3e}"
                )

        return loss
    
    def closureMSE(self):
        #to run in the training loop

        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        loss,l1,l2,l3=self.mse_loss(xy_actual,uvp_act)

        loss.backward()
        self.losses["bc1"].append(l1.detach().cpu().item())
        self.losses["bc2"].append(l2.detach().cpu().item())
        self.losses["out"].append(l3.detach().cpu().item())
        
        self.iter +=  1

        if(self.iter%200 == 0):
            print(f" It: {self.iter} Loss: {loss.item():.5e}"
                )

        return loss

    

if __name__  ==  "__main__":
    pinn = PINN()
    for i in range(3000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    pinn.mse_loss(xy_actual,uvp_act)
    torch.save(pinn.net.state_dict(), "Param.pt")
    plotLoss(pinn.losses, "LossCurve.png", ["BC1","BC2","OUT","PDE"])