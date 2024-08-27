import torch 
from torch import nn
import numpy as np
import torch.optim as optim
import torch.autograd as ag
import matplotlib.pyplot as plt
from numpy import random

torch.autograd.set_detect_anomaly(True)

class Physics_Network(torch.nn.Module):
    #intitialize
    #nodes are how many nodes the cst data has 
    def __init__(self):
        super(Physics_Network,self).__init__()
        #subject to change 
        #x,y cords for cst data + rest of inputs 
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,3)
        self.drop = nn.Dropout(0.1)
        self.activation = nn.Tanh()

    #forward pass
    #TODO: maybe try different activation/order 
    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        return(x)


def print_output(outputs):
    print('pressure: ',outputs[0].item())
    print('velocity vector: [',outputs[1].item(),',',outputs[2].item(),']')

def graph(inputs,outputs):
    plt.plot(inputs[0].item(),inputs[1].item())
    plt.show()

def physics_loss(model, x,y, rho=1):
    inputs = torch.tensor([[x,y]],requires_grad=True)
    ans=model(inputs)
    p=ans[0,0]
    u=ans[0,1]
    v=ans[0,2]
    
    #derivatives
    u_=torch.autograd.grad(u,inputs,grad_outputs=torch.ones_like(u),
                            create_graph=True)[0]
    v_=torch.autograd.grad(v,inputs,grad_outputs=torch.ones_like(v),
                            create_graph=True)[0]
    p_=torch.autograd.grad(p,inputs,grad_outputs=torch.ones_like(p),
                            create_graph=True)[0]
    print(u_,v_,p_)

    u_x,u_y = u_[:,0],u_[:,1]
    v_x,v_y = v_[:,0],v_[:,1]
    p_x,p_y = p_[:,0],p_[:,1]

    #continuity 
    loss_c=torch.mean(((u_x+v_y)-0)**2)

    #xmom
    loss_x=torch.mean(((u*u_x+v*u_y+(1/rho)*p_x)-0)**2)

    #ymom
    loss_y=torch.mean(((u*v_x+v*v_y+(1/rho)*p_y)**2))
    return loss_c+loss_x+loss_y



#define network
model = Physics_Network()

#define boundry points
blist=[]
for y in range(10):
    blist.append((model(0,y)-0)**2)
b1=torch.mean(blist)

blist=[]
for y in range(10):
    blist.append((model(10,y)-10)**2)
b2=torch.mean(blist)


optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)#learning rate


print("loss",physics_loss(model,.97,.45))


for i in range(1000):
    optimizer.zero_grad()

    blist=[]
    for y in range(10):
        blist.append((model(0,y)-0)**2)
    b1=torch.mean(blist)

    blist=[]
    for y in range(10):
        blist.append((model(10,y)-10)**2)
    b2=torch.mean(blist)

    physics_loss