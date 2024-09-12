import torch
import vtk

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

class VTU():

    def read_data(path):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName("flow.vtu")
        reader.Update()

        # Get the data from the reader
        data = reader.GetOutput()

        points = data.GetPoints()
        num_points = points.GetNumberOfPoints()

        pressure = data.GetPointData().GetArray("Pressure")

        velocity = data.GetPointData().GetArray("Velocity")
        points_array=[]
        pressure_array=[]
        velocity_array=[]
        for i in range(num_points):
            points_array.append(list(points.GetPoint(i)))
            pressure_array.append(pressure.GetValue(i))
            velocity_array.append(list(velocity.GetTuple(i)))

        reduced_points= []
        reduced_pressure= []
        reduced_v= []
        reduced_u = []
        for i in range(num_points):
            point=points_array[i]
            if -.5<point[0]<2 and -.5<point[1]<.5:
                vel=velocity_array[i]
                point.pop(2)
                reduced_points.append(point)
                reduced_pressure.append(pressure_array[i])
                reduced_v.append(vel[1])
                reduced_u.append(vel[0])

        tpoints=torch.tensor(reduced_points)
        tpressure=torch.tensor(reduced_pressure)
        maxp=torch.max(tpressure)
        tpressure=(tpressure-maxp/2)/tpressure
        tu=torch.tensor(reduced_u)
        maxu=torch.max(tu)
        tu=(tu-maxu/2)/maxu
        tv=torch.tensor(reduced_v)
        maxv=torch.max(tv)
        tv=(tv-maxv/2)/maxv
        return tpoints,[tu,tv,tpressure]
    
