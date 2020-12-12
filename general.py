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
  
  
  
