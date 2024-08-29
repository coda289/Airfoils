import torch
import numpy as np

class CSV():
    def read_data(path):
        with open(path,'r') as f:
            fileLines = f.readlines()
        label=fileLines[0].split(',')
        fileLines.pop(0)
        data=[[],[],[],[],[],[],[],[],[]]
        for line in fileLines:
            point=line.split(',')
            point[len(point)-1]= point[len(point)-1][0:len(point[1])-2]
            for i in range(len(point)):
                point[i]=float(point[i])
                data[i].append(point[i])
        return data
    
    def tsplit_data(data):
        p=data[0]
        u=data[1] 
        v=data[2] 
        pc=data[4] 
        d=data[5]
        x=data[6] 
        y=data[7] 
        p=torch.tensor(p,dtype=torch.float32,requires_grad=True)
        u=torch.tensor(u,dtype=torch.float32,requires_grad=True)
        v=torch.tensor(v,dtype=torch.float32,requires_grad=True)
        x=torch.tensor(x,dtype=torch.float32,requires_grad=True)
        y=torch.tensor(y,dtype=torch.float32,requires_grad=True)
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=-1)
        xy=torch.tensor(xy,dtype=torch.float32,requires_grad=True)
        return xy,u,v,p

if __name__=='__main__':
    data=CSV.read_data('flow.csv')
    CSV.tsplit_data(data)
