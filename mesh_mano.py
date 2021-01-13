import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

############################################################################


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


##################################
with open('tools_ess/Regressor_sphereToNYUjoints.pickle', 'rb') as f:
    Regressor = pickle.load(f)

def batch_sphere_to_joints2(spheres_batch,weights):
    # spheres is a tensor of shape (batch,n,3)
    # the output will be a tensor of shape (batch,m,3)

    return torch.matmul(weights.double(),spheres_batch.double())
#####################################




def from_AtoBIV(a,b,ratio=1):
    # a and b: each a tensor of size (3)
    # this function returns a rotation matrix R such that a*R will align with b if ratio=1
    # basically rotates the a by theta*ratio degree, where theta is the angle between a and b
    normal=torch.cross(a,b)/torch.norm(torch.cross(a,b))
    normal=normal.view(1,3)
    cos0=torch.dot(a,b)/(torch.norm(a)*torch.norm(b))
    
    theta=np.arccos(cos0-1e-6)*ratio#+np.pi#+1*np.pi
    R=batch_rodrigues(theta*normal).view(3,3)
    return torch.transpose(R,0,1),-theta*normal


def Inverse_Kinematic(Joints,scale=1,pca_use=False,ncomps=6,address="mano/models/MANO_RIGHT.pkl",flat_hand_mean=True):
    # Joints is a tensor of size (k,3)
    # Joints should be in the original order
    
    data = pickle.load(open(address, 'rb'), encoding='latin1')
    restPose=torch.from_numpy(data["J"]*1000*scale).float()
    
    lev1_idxs = [1, 4, 7, 10, 13]
    lev2_idxs = [2, 5, 8, 11, 14]
    lev3_idxs = [3, 6, 9, 12, 15]
    levels=[lev1_idxs,lev2_idxs,lev3_idxs]
    
    # compute the root rotation matrix and axis-angle
    As=[]
    Bs=[]
    for g in lev1_idxs:
        a=(restPose[g]-restPose[0])
        b=Joints[g]-Joints[0]
        As.append(a)
        Bs.append(b)

    A=torch.cat([As[0],As[1],As[2]]).view(3,3)

    R=[]
    for i in range(3):
        B=torch.stack([Bs[0][i],Bs[1][i],Bs[2][i]]).view(3,1)
        r=torch.inverse(A)@B
        R.append(r)

    R_root=torch.stack(R).view(3,3)
    ax_root=torch.from_numpy(Rotation.from_matrix(R_root).as_rotvec())

    my_pose=torch.zeros(15,3)
    my_R=torch.zeros(1,15,3,3)
    R_accum=torch.stack([torch.eye(3) for i in range(16)])
    R_accum[lev1_idxs]=R_root

    for lvl in range(2):
            parents_lvl=levels[lvl]
            children_lvl=levels[lvl+1]
            for i in range(5):
                vec1=restPose[children_lvl[i]]-restPose[parents_lvl[i]]
                vec2=Joints[children_lvl[i]]-Joints[parents_lvl[i]]
#                 R,axs2=from_AtoBIV(vec2,R_accum[parents_lvl[i]]@vec1)
                R,axs2=from_AtoBIV(R_accum[parents_lvl[i]].T@vec2,vec1)

                
                my_R[0,parents_lvl[i]-1]=R
                R_accum[children_lvl[i]]=R_accum[parents_lvl[i]]@R
                my_pose[parents_lvl[i]-1,:]=axs2
      
        
    my_pose=my_pose.view(1,-1)
    
    if pca_use:
        hands_components=torch.from_numpy(data["hands_components"]).float()
        hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else data['hands_mean']
        hands_mean = torch.Tensor(hands_mean).unsqueeze(0).float()
        selected_components = hands_components[:ncomps]
        my_pose=(selected_components@(my_pose-hands_mean).T).T #(1,k)
        

        
    pose=torch.cat([ax_root.view(1,-1),my_pose],dim=1)
    return pose


# from_AtoBIV(b,a)
# r@a=b
