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
    def __init__(self,nodes=151):
        super(Physics_Network,self).__init__()
        #subject to change 
        #x,y cords for cst data + rest of inputs 
        self.fc1 = nn.Linear(nodes*2+4,nodes)
        self.fc2 = nn.Linear(nodes,20)
        self.fc3 = nn.Linear(20,10)
        self.fc4 = nn.Linear(10,3)
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
        x = self.fc4(x)
        x = self.activation(x)
        return(x)

#this function formats the data into a tensor 
# each element including each x and y cord of the cst 
# is one element in the tensor
#  then it is scaled between -1 and 1
def inputs_scaled(x_cord,y_cord,aoa,mach,cst):
    flat=np.concatenate(cst).tolist()
    maxCST = max(flat)
    maxO=max(x_cord,y_cord,aoa,mach,maxCST)
    list = [x_cord,y_cord,aoa,mach]+flat
    inputs=torch.tensor(list,requires_grad=True)
    inputs=inputs-maxO/2
    inputs=inputs/(maxO/2)
    return inputs , maxO

def outputs_rescale(outputs,max):
    outputs=outputs*max/2
    outputs=outputs+max/2
    return outputs

def print_output(outputs):
    print('pressure: ',outputs[0].item())
    print('velocity vector: [',outputs[1].item(),',',outputs[2].item(),']')

def graph(inputs,outputs):
    plt.plot(inputs[0].item(),inputs[1].item())
    plt.show()

def physics_loss(model, inputs,rho=1):
    inputs.requires_grad_(True)
    #variables
    ans=model(inputs)
    p=ans[0:1]
    u=ans[1:2]
    v=ans[2:3]
    
    #derivatives
    u_xy=torch.autograd.grad(u,inputs,grad_outputs=torch.ones_like(u),
                            create_graph=True)[0]
    u_y=u_xy[1]
    print(u_xy)
    v_y=torch.autograd.grad(v,inputs,grad_outputs=torch.ones_like(v),
                            create_graph=True)[1]
    v_x=torch.autograd.grad(v,inputs,grad_outputs=torch.ones_like(v),
                            create_graph=True)[0]
    p_x=torch.autograd.grad(p,inputs,grad_outputs=torch.ones_like(p),
                            create_graph=True)[0]
    p_y=torch.autograd.grad(p,inputs,grad_outputs=torch.ones_like(p),
                            create_graph=True)[1]

    #continuity 
    loss_c=torch.mean(((u_x+v_y)-0)**2)

    #xmom
    loss_x=torch.mean(((u*u_x+v*u_y+(1/rho)*p_x)-0)**2)

    #ymom
    loss_y=torch.mean(((u*v_x+v*v_y+(1/rho)*p_y)))
    return loss_c+loss_x+loss_y

#define network
model = Physics_Network(nodes=25)

#define boundry points

optimizer = optim.Adam(model.parameters())#learning rate


x,max=inputs_scaled(5,8,15,.87,random.rand(25,2))
outputs=model(x)
out_scale=outputs_rescale(outputs,max)
print_output(out_scale)

print(physics_loss(model,x))
