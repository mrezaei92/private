import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def interpolate(ann,indx=[0,1]):
    mid=(ann[indx[0]]+ann[indx[1]])/2
    c1=(mid+ann[indx[1]])/2
    c2=(ann[indx[0]]+mid)/2
    rad=np.linalg.norm((ann[indx[0]]-ann[indx[1]]))/4
    return c1,c2,rad
  
  
def compute_Spheres(mesh_v,H_joints):
    # mesh_v is a tensor of size (n,3) where n is the number of vertices
    # H_joints is a tensor of size (m,3) where m is the number of hand joints
    # returns a torch tensor of size (41,3) and a tensor of size(41,1), which is their respective radius
    dic={}
    dic[63]=[145]
    dic[271]=[206]
    dic[776]=[182]
    dic[2]=[158]
    dic[66]=[21,22]
    dic[180]=[73]
    dic[218]=[26,265]
    dic[179]=[244,130]
    dic[40]=[229,232]
    dic[191]=[37,111]
    dic[205]=[32,33]

    pairs=[[1,2],[2,3],[3,4],[5,6],[7,6],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    interpolated=[]
    for pp in pairs:
        c1,c2,rad=interpolate(H_joints,pp)
        interpolated.append(c1)
        interpolated.append(c2)

    for k in dic.keys():
        a1=mesh_v[k]
        if len(dic[k])==1:
            a2=mesh_v[dic[k]]
        else:
            a2=(mesh_v[dic[k][0]]+mesh_v[dic[k][1]])/2

        interpolated.append(torch.squeeze((a1+a2)/2))


    spheres=np.array(torch.stack(interpolated))
    RAD=[8,11,10,9,9.5,9.5,  9,12,8,8,8,8,  9,13,8,8,8,8,  8,14,8,8,8,8,  7,8,7,7,7,7,  16,14,14,15,16,15, 19,14,14,14,14]
    Radius=torch.tensor(RAD).reshape(-1,1)
    return spheres,Radius
  
    
    
    
def sphere_to_joints(spheres):
    # spheres is a tensor of shape (n,3)
    # the output will be a tensor of shape (m,3)
    jointss=[]
    p=[28,[24,27],22,[18,21],16,[12,15],10,[6,8],4,2,0,40,[38,39],[31,36]]
    for i in range(len(p)):
        if type(p[i])!=type([1]):
            jointss.append(spheres[p[i]])
        else:
            n=0;
            summ=p[i][0];
            for j in range(len(p[i])):
                n=n+1;
                summ=summ+spheres[p[i][j]]
                
            summ=summ-p[i][0]
            jointss.append(summ/n)
    return torch.stack(jointss)




def get_NYU_compatible_joints(mesh_verts,joints,selected_joints=[20,18,16,14,12,10,8,6,4,3],selected_verts=[[231,7] ,[44,85], [37,190],67]):#[44,37,67]):
    # mesh verts and joints of of size (m,3) and (n,3) respectively
    final=joints[:,selected_joints]
    batch_size=mesh_verts.shape[0]
    for element in selected_verts:
        if type(element)==int:
            temp=mesh_verts[:,element].reshape(batch_size,1,3)
        else:
            temp=(mesh_verts[:,element[0]].reshape(batch_size,1,3)+mesh_verts[:,element[1]].reshape(batch_size,1,3))/2
            
        final=torch.cat([final,temp],dim=1)
    return final 







