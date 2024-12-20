
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from read_data import CSV, DAT
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from torch.autograd import grad

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#not really neccesary 
torch.manual_seed(1234)
np.random.seed(1234)

#The min and max values of x and y
y_min = -50
y_max = 50
x_min = -50
x_max = 50

Re = 200

#Amount of points taken along boundry condition
bound_num = 50
#Amount of points to evaluate each step
points_num = 1000

#path to airfoil data
path_to_points = 'ah79100b.dat'

validation_data = CSV.read_data('flow.csv')
uvp_list, xy_actual = CSV.tsplit_data(validation_data)
uvp_act = uvp_list[2:5]

#turns the data into an array
airfoil_points = DAT.read_data(path_to_points)

#turns the array into a tensor
t_airfoil_points = torch.tensor(
    airfoil_points,dtype=torch.float32,requires_grad=True)

#returns the points from the boundry condition
#returns random points for the network to use
#(in the range but out of the airfoil)
def make_graph(data, xs_min=-.2, xs_max=1.2, ys_min=-.5, ys_max=.5):
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
    c1 = torch.stack((front_wallx, front_wally), dim=1)
    c2 = torch.stack((top_wallx, top_wally), dim=1)
    c3 = torch.stack((bottom_wallx, bottom_wally), dim=1)
    bc1 = torch.concatenate([c1,c2,c3])
    bc1.clone().detach().requires_grad_(True)

    #make random 
    x = []
    y = []
    xy = []
    polygon = Polygon(data)
    #generate and validate data
    while len(xy)<points_num/2:
        xp = np.random.uniform(x_min, x_max)
        yp = np.random.uniform(y_min, y_max)
        point = Point(xp,yp)
        if not polygon.contains(point):
            x.append(xp)
            y.append(yp)
            xy.append([xp, yp])
    while len(xy)<points_num:
        xp = np.random.uniform(xs_min,xs_max)
        yp = np.random.uniform(ys_min, ys_max)
        point = Point(xp,yp)
        if not polygon.contains(point):
            x.append(xp)
            y.append(yp)
            xy.append([xp, yp])
    rand_points = torch.tensor(xy,dtype=torch.float32)
    plt.scatter(x,y,marker='o',c='red')
    return bc1,rand_points

bc1,rand_points = make_graph(airfoil_points)

#returns normal vectors of each point on the airofoil
def normal_data(data):

    previous = None
    next = data[2]
    normals = []

    for i in range(len(data)):
        pair = data[i] 
        plt.plot(pair[0],pair[1],'o')
        if previous !=  None:
            if next !=  None:
                normalx = previous[0]-next[0]
                normaly = previous[1]-next[1]
                #plt.quiver(pair[0],pair[1],normaly,normalx)
            else:
                next = data[0]
                normalx = previous[0]-next[0]
                normaly = previous[1]-next[1]
                #plt.quiver(pair[0],pair[1],normaly,normalx)
        else:
            normalx = -1
            normaly = 0
            #plt.quiver(pair[0],pair[1],normaly,normalx)

        previous = pair 
        if i+2<len(data):
            next = data[i+2]
        else:
            next = None
        length=np.sqrt(normalx**2+normaly**2)
        normals.append([normaly/length,normalx/length])
    normals=np.array(normals)  
    normals = torch.tensor(normals,requires_grad=True)
    return normals

n_data = normal_data(airfoil_points)

#plots the loss
def plotLoss(losses_dict, path, info=["I.C.", "B.C.", "P.D.E."]):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(3), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)

#initalizes the weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


#layer class
class layer(nn.Module):

    #layer initialize
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    #forward pass
    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

#Model class
class DNN(nn.Module):

    #initilize
    def __init__(self, dim_in, dim_out, n_layer, n_node, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.net.apply(weights_init)

    #forward pass
    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        return out

#
class PINN:

#TODO change to represent real life
    U_inf = 1 
    rho = 1
    AoA = 0
    
    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=6, n_layer=10, n_node=128).to(
            device
        )

        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=10000,
            max_eval=10000,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        #self.losses = {"l1": [], "l2": [], "l3": []}
        #self.losses = {"bc1": [], "bc2": [], "pde": [],"l1": [], "l2": [], "l3": []}
        self.losses = {"bc1": [], "bc2": [], "pde": [],"mse":[]}
        self.lw = [1,1,1.5]
        self.iter = 0

    def predict(self, X):
        out = self.net(X)

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        sig_xx = out[:, 3:4]
        sig_xy = out[:, 4:5]
        sig_yy = out[:, 5:6]

        return u, v, p, sig_xx, sig_xy, sig_yy
    
    def bc_loss1(self, X,U_inf=U_inf,AoA=AoA):
        u, v = self.predict(X)[0:2]
        
        mse_bc = torch.mean(torch.square(u - U_inf*np.cos(AoA))) + torch.mean(
            torch.square(v - U_inf*np.sin(AoA))
        )

        return mse_bc
    
    def bc_loss2(self,X,n_data):
        u,v = self.predict(X)[0:2]
        uv = torch.concatenate([u,v], axis=1)
        
        mse_bc = torch.mean(
            torch.square(
                torch.sum(uv*n_data,dim=1)
            )
        )
        return mse_bc
    
    def pde_loss(self, X,rho=rho):
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
        v_t = u_out[:, 2:3]

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
        X = X.clone()
        #X.requires_grad = True
        pred_u, pred_v, pred_p,s1,s2,s3 = self.predict(X)
        
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
        

        return loss


    def closure(self):

        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc1 = self.bc_loss1(bc1)
        mse_bc2 = self.bc_loss2(t_airfoil_points,n_data)
        mse_pde = self.pde_loss(rand_points)

        #mse_loss= self.mse_loss(xy_actual,uvp_act)
        #self.loss=loss
        #self.losses["l1"].append(l1.detach().cpu().item())
        #self.losses["l2"].append(l2.detach().cpu().item())
        #self.losses["l3"].append(l3.detach().cpu().item())

        loss = mse_bc1 + mse_bc2 + mse_pde

        loss.backward()

        self.losses["bc1"].append(mse_bc1.detach().cpu().item())
        self.losses["bc2"].append(mse_bc2.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        
        self.iter +=  1

        if(self.iter%200 == 0):
            print(f" It: {self.iter} Loss: {loss.item():.4e} BC1: {mse_bc1.item():.3e} BC2: {mse_bc2.item():.3e} pde: {mse_pde.item():.3e}"
                )

        #if(self.iter%200==0):
        #   print(self.iter,"loss:",loss," loss u:",l1," loss v:",l2," loss p:",l3)

        return loss
    

    def closure1(self):

        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        loss, l1,l2,l3= self.mse_loss(xy_actual,uvp_act)
        self.loss=loss
        self.losses["l1"].append(l1.detach().cpu().item())
        self.losses["l2"].append(l2.detach().cpu().item())
        self.losses["l3"].append(l3.detach().cpu().item())
        loss.backward()
        
        self.iter +=  1

        if(self.iter%200==0):
            print(self.iter,"loss:",loss," loss u:",l1," loss v:",l2," loss p:",l3)
            print(pinn.pde_loss(xy_actual))
        return loss
    
    def closure2(self):

        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        

        mse_pde = self.pde_loss(rand_points)

        loss = mse_pde

        loss.backward()

        self.losses["pde"].append(mse_pde.detach().cpu().item())
        
        self.iter +=  1

        if(self.iter%200 == 0):
            print(f" It: {self.iter} Loss: {loss.item():.5e} pde: {mse_pde.item():.3e}"
                )


        return loss

    def closure3(self):

        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc1 = self.bc_loss1(bc1)
        mse_bc2 = self.bc_loss2(t_airfoil_points,n_data)

        loss = mse_bc1 + mse_bc2

        loss.backward()

        self.losses["bc1"].append(mse_bc1.detach().cpu().item())
        self.losses["bc2"].append(mse_bc2.detach().cpu().item())
        
        self.iter +=  1

        if(self.iter%200 == 0):
            print(f" It: {self.iter} Loss: {loss.item():.5e} BC1: {mse_bc1.item():.3e} BC2: {mse_bc2.item():.3e} "
                )

        

        return loss

    def validate(self,X,actual):
        X = X.clone()
        #X.requires_grad = True
        pred_u, pred_v, pred_p = self.predict(X)
        
        loss_func = nn.MSELoss()
        u,v,p = actual[0], actual[1], actual[2]
        u = u.view(-1,1)
        v = v.view(-1,1)
        p = p.view(-1,1)

        loss1 = loss_func(pred_u*100,u)
        loss2 = loss_func(pred_v*100,v)
        loss3 = loss_func(pred_p*100,p)    

        loss=loss1+loss2+loss3
        
        print('u loss',torch.mean(loss1))
        print('v loss',torch.mean(loss2))
        print('p loss',torch.mean(loss3))
        print('total',loss)
        

        return loss,loss1,loss2,loss3

    

if __name__  ==  "__main__":
    pinn = PINN()
    for i in range(2000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    pinn.mse_loss(xy_actual,uvp_act)
    torch.save(pinn.net.state_dict(), "c:/Users/DakotaBarnhardt/Downloads/Airfoils/Param.pt")
    plotLoss(pinn.losses, "c:/Users/DakotaBarnhardt/Downloads/Airfoils/LossCurve.png", ["BC1","BC2","PDE"])


'''
if __name__  ==  "__main__":
    pinn = PINN()
    for i in range(3000):
        pinn.closure1()
        pinn.adam.step()
    #pinn.lbfgs.step(pinn.closure1)
    torch.save(pinn.net.state_dict(), "c:/Users/DakotaBarnhardt/Downloads/Airfoils/Param1.pt")
    plotLoss(pinn.losses, "c:/Users/DakotaBarnhardt/Downloads/Airfoils/LossCurve1.png", ["l1","l2","l3"])



    for i in range(2000):
        pinn.closure2()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure2)
    torch.save(pinn.net.state_dict(), "c:/Users/DakotaBarnhardt/Downloads/Airfoils/Param2.pt")
    plotLoss(pinn.losses, "c:/Users/DakotaBarnhardt/Downloads/Airfoils/LossCurve2.png", ["PDE"])


    for i in range(2000):
        pinn.closure3()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure3)
    torch.save(pinn.net.state_dict(), "c:/Users/DakotaBarnhardt/Downloads/Airfoils/Param3.pt")
    plotLoss(pinn.losses, "c:/Users/DakotaBarnhardt/Downloads/Airfoils/LossCurve3.png", ["BC1","BC2"])


    pinn.mse_loss(xy_actual,uvp_act)
    torch.save(pinn.net.state_dict(), "c:/Users/DakotaBarnhardt/Downloads/Airfoils/Param.pt")
    plotLoss(pinn.losses, "c:/Users/DakotaBarnhardt/Downloads/Airfoils/LossCurve.png", ["BC1","BC2","PDE"])

    def pde_loss(self, X,rho=rho):
        X = X.clone()
        X.requires_grad = True

        u, v, p = self.predict(X)
        
        du_out = grad(u.sum(), X, create_graph=True)[0]
        dv_out = grad(v.sum(), X, create_graph=True)[0]
        dp_out = grad(p.sum(), X, create_graph=True)[0]

        dudx = du_out[:, 0:1]
        dudy = du_out[:, 1:2]
        dvdx = dv_out[:, 0:1]
        dvdy = dv_out[:, 1:2]
        dpdx = dp_out[:, 0:1]
        dpdy = dp_out[:, 1:2]
    

        fc = dudx + dvdy
        fx = (u*dudx + v*dudy)+(1/rho)*dpdx
        fy = (u*dvdx + v*dvdy) +(1/rho)*dpdy
  

        mse_fc = torch.mean(torch.square(fc))
        mse_fx = torch.mean(torch.square(fx))
        mse_fy = torch.mean(torch.square(fy))

        mse_pde = mse_fc + mse_fx + mse_fy

        return mse_pde
'''
    






