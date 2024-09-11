import torch

class CSV():
    #to interpret csv data 

    def read_data(path,x_min=-1,x_max=2,y_min=-.5,y_max=.5):
        #turn from a csv file to a list of lists

        with open(path,'r') as f:
            fileLines = f.readlines()
        fileLines.pop(0)

        data=[[],[],[],[],[],[],[],[],[]]
        for line in fileLines:
            point = line.split(',')
            point[len(point)-1] = point[len(point)-1][0:len(point[1])-2]
            for i in range(len(point)):
                point[i]=float(point[i])
            if x_min<point[6] and point[6]<x_max and y_min<point[7] and point[7]<y_max:
                for i in range(len(point)):
                    data[i].append(point[i])

        return data
    
    
    def tsplit_data(data):
        #split the data into tensors

        p = data[0]
        u = data[1]
        v = data[2]
        x = data[6]
        y = data[7]
        p = torch.tensor(p,dtype=torch.float32)
        u = torch.tensor(u,dtype=torch.float32)
        v = torch.tensor(v,dtype=torch.float32)
        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.float32)
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=-1)
        xy.clone().detach().requires_grad_(True)
        return [x,y,u,v,p],xy

class DAT():
    #to read DAT files 

    def read_data(path):
        #takes a DAT file and return a list of points 

        with open(path,'r') as f:
            fileLines = f.readlines()
        fileLines.pop(0)

        airfoilPoints = []
        for line in fileLines:
            point = line.split(' ')
            while point.__contains__(''):
                point.remove('')
            point[0] = float(point[0])
            point[1] = float(point[1][0:len(point[1])-2])
            airfoilPoints.append(point)

        return airfoilPoints
