import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NeuralRenderer(torch.nn.Module):
    def __init__(self,Radius=None,num_circles=36,dimension=128,Dfar=512):
        super(NeuralRenderer, self).__init__()
        # This module is a Depth Renderer. It takes a uvd of shape (batch,num_circles,3) and renders 
        # the corresponding depth map. uvd is the u,v coordinate and depth of each circle. The rendering
        # is done through orthographic projection. Radius is a tensor of shape (num_circles,1), where each
        # element represents the radius of the corresponding circle.
        # Dfar denotes the non-hand value in the the output depth map
        # the return value will be a tensor of shape (batch,1,dimension,dimension)

        self.Dfar=Dfar
        self.dimension=dimension
        self.num_circles=num_circles
        if Radius is None:
            self.Radius=torch.ones(num_circles,1).double()*5.8
        else:
            self.Radius=Radius

        X=torch.unsqueeze( torch.squeeze(GetValuesX(dimension=self.dimension)),dim=0).double()
        Y=torch.unsqueeze( torch.squeeze(GetValuesY(dimension=self.dimension)), dim=0).double()
        uv=torch.unsqueeze(torch.cat([X,Y],dim=0),dim=0)
        self.UV=uv
        for i in range(num_circles-1):
            self.UV=torch.cat([self.UV,uv],dim=0)

        self.UV=nn.Parameter(self.UV,requires_grad=False)
        self.Radius=nn.Parameter(self.Radius,requires_grad=False)
        #self.Radius.requires_grad = False

    def forward(self, uvd):
        num_batch=uvd.shape[0]
        UV=torch.unsqueeze(self.UV,dim=0).repeat(num_batch,1,1,1)

        f=torch.unsqueeze(uvd[:,:,0:2],dim=-1)#.double()
        f=f.repeat(1,1,1,self.dimension*self.dimension)#.double()
        D=uvd[:,:,2].view(num_batch,self.num_circles,1)#.double()

        res=(UV-f)**2+1e-12
        res=torch.sqrt(res.sum(dim=2))
        ind=res>=self.Radius;#print(ind.shape)
        mask1=1-ind.double();
        mask2=ind.double()*self.Dfar
        #res[ind]=0
        res=res*mask1
        res=D-torch.sqrt( (self.Radius**2)-(res**2) )
        #res[ind]=self.Dfar
        res=res*mask1+mask2
        rend=torch.min(res,dim=1)[0].view(num_batch,self.dimension,self.dimension)
        rend=rend.unsqueeze(1)
        return rend#,ind


class TemporalLoss(nn.Module):
    def __init__(self,cost):
        super(TemporalLoss, self).__init__()
        self.costFunction=cost
    def forward(self, x):
        # x is a tensor of size(batch,num_keypoints,3), Remember, each consecutive elements correspond to consecutive frames
        # e.g. x[0] and x[1] are supposed to be predictions for two consecutive frames
        b_size=x.shape[0]
        first=x[[i for i in range(0,b_size,2)],:]
        second=x[[i for i in range(1,b_size,2)],:]
        return self.costFunction(first,second)


class D2MLoss(torch.nn.Module):
    def __init__(self,rads,num_circles=41,dimension=128,max_depth=256):
        super(D2MLoss, self).__init__()
        # This module computer the data to model loss of Wan's paper
        # rads should be a torch of size (num_circles,1) #determines the radious of each circle
        # num_circle is the number of circle our model predicts
        # dimension is the dimension of the input depth map, e.g 128 * 128

        self.x=torch.unsqueeze( torch.squeeze(GetValuesX(dimension=128)),dim=0)
        self.y=torch.unsqueeze( torch.squeeze(GetValuesY(dimension=128)), dim=0)
        self.rads=rads
        self.num_circles=num_circles
        self.max_depth=max_depth
        self.x=nn.Parameter(self.x,requires_grad=False)
        self.y=nn.Parameter(self.y,requires_grad=False)
        self.rads=nn.Parameter(self.rads,requires_grad=False)



    def forward(self, uvd,img):
        #uvd is a torch of size (num_batch,num_circle,3), which corresponds to x,y and z of the respective circles
        #img is the depth map, a torch of size (num_batch,1, dimension, dimension)
        num_batch=uvd.shape[0]
        
        mask=((img!=self.max_depth).double()).transpose(2,3).reshape(num_batch,-1) # because of the column-wise stacking
        X=self.x.repeat(num_batch,self.num_circles,1).unsqueeze(-1)#.double()
        Y=self.y.repeat(num_batch,self.num_circles,1).unsqueeze(-1)#.double()
        Ds=img.transpose(2,3).reshape(num_batch,1,-1).repeat(1,self.num_circles,1).unsqueeze(-1).double()
        UVD=torch.cat([Y,X,Ds],dim=-1).double()
        circles=uvd.unsqueeze(-1).double().transpose(2,3)
        radius=self.rads.repeat(num_batch,1,1)
        res=((UVD-circles)**2)
        res=torch.sqrt(res.sum(dim=-1))
        res=torch.abs(res-radius)
        res=torch.min(res,dim=1)[0]*mask
        #print(res.shape,mask.shape)
        final=torch.sum(res,dim=1)

        return final


class BoneLengthLoss(torch.nn.Module):
    def __init__(self,factor_normalization,
                 bones=[(4,5),(2,3),(0,1),(10,11),(8,9),(6,7),(16,17),(14,15),(12,13),
                        (22,23),(20,21),(18,19),(29,28),(26,27),(24,25),(7,30),(13,31),(19,34),(25,32),
                         (34,36),(36,33),(35,37),(36,39),(40,37),(1,38)  ],
                 mins=[ 9,5,7,3 ,  3,8  , 4,  3, 7, 4, 4. ,  6.,  1.,  1.,  2.,16. ,13. , 28. , 9, 21, 21, 14, 24, 14, 12. ],
                 maxs=[26,22,24,20,20.,25.,21.,20.,24.,21.,21.,23., 18.,18., 19., 33., 30., 45., 26., 38.,38., 31., 41.,31.,29.]):
        
        # bones contains the child and parent indeceis, min and max contain the min value
        # and max value for the corresponding bonelengths respectively
        super(BoneLengthLoss, self).__init__()
        self.mins=torch.from_numpy(np.array(mins).reshape(1,-1)*factor_normalization)
        self.maxs=torch.from_numpy(np.array(maxs).reshape(1,-1)*factor_normalization)
        self.mins=nn.Parameter(self.mins,requires_grad=False)
        self.maxs=nn.Parameter(self.maxs,requires_grad=False)


        self.childs=[]
        self.parens=[]
        for b in range(len(bones)):
            self.childs.append(bones[b][0])
            self.parens.append(bones[b][1])

        self.zerros=nn.Parameter(torch.zeros(32,len(self.childs)).double(),requires_grad=False)



    def forward(self, xyz):
        # the input should be of shape (batch,num_keypoints,3)
        
        a=torch.sum( (xyz[:,self.childs]-xyz[:,self.parens])**2 ,dim=2 )
        a=torch.sqrt(a)
        
        zeros=self.zerros[0:xyz.shape[0],:]
        maxs=self.maxs.repeat(xyz.shape[0],1)
        mins=self.mins.repeat(xyz.shape[0],1)
        
        maxs=a-maxs
        mins=mins-a
        min_losses=torch.max(mins,zeros)**2
        max_losses=torch.max(maxs,zeros)**2
        return torch.mean(min_losses+max_losses)



class AdaptiveSpatialSoftmaxLayer(nn.Module):
    def __init__(self,spread=None,train=True,num_channel=14):
        super(AdaptiveSpatialSoftmaxLayer, self).__init__()
        # spread should be a torch tensor of size (1,num_chanel,1)
        # the softmax is applied over spatial dimensions 
        # train determines whether you would like to train the spread parameters as well
        #self.SpacialSoftmax = nn.Softmax(dim=2)
        if train:
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #print(device)
            self.spread=nn.Parameter(torch.ones(1,num_channel,1))#.double().to(device))
            #self.spread.requires_grad=True
        else:
            self.spread=spread.double()#.to(device)
            #if spread is not None:
            #    self.spread.requires_grad=False



    def forward(self, x):
        # the input is a tensor of shape (batch,num_channel,height,width)
        SpacialSoftmax = nn.Softmax(dim=2)
        num_batch=x.shape[0]
        num_channel=x.shape[1]
        height=x.shape[2]
        width=x.shape[3]
        inp=x.view(num_batch,num_channel,-1)
        #if self.spread is not None:
        res=torch.mul(inp,self.spread)
        res=SpacialSoftmax(inp)
        
        return res.reshape(num_batch,num_channel,height,width)







########################################################################################################

def resize_heatmap(HP,scale_factor=2):
        # HP is a heatmap of the shape (batch,num_joints,w,w)
        # returns a resized hatmap of the shape (batch,num_joints, w*scale_factor, w*scale_factor)
    
    from skimage import transform
    
    new_w=np.int16(np.round(HP.shape[3]*scale_factor))
    new_h=np.int16(np.round(HP.shape[2]*scale_factor))
    
    result=np.zeros((HP.shape[0],HP.shape[1],new_h,new_w))
    for b in range(HP.shape[0]):
        for joint in range(HP.shape[1]):
            result[b,joint,:,:]=transform.resize(HP[b,joint,:,:], (new_h, new_w))
    
    return result



def draw_gussian(img, pt, sigma):
    # img should be an array of the image size initialized with zeros
    # pt is a list, [x,y], which is the point where you want to center a guassian peak on
    # reutrns a heatmap of shape (num_joints, w,w) , where w is the dimension of the image
    
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def draw_heatmap(uvd,w,sigma=1):
    # uvd is the 2D joint location of images, of the shape (num_joints,2), where uvd[:,0] are Xs and uvd[:,0] are Ys
    # w is the dimension of the output heatmap
    # returns a heatmap of shape (num_joints,w,w)
    
    heatmap=np.zeros((uvd.shape[0],w,w))
    for joint in range(uvd.shape[0]):
        x=uvd[joint,0]
        y=uvd[joint,1]
        if x>=0 and y>=0 and x<w and y<w:
            heatmap[joint,:,:]=draw_gussian(np.squeeze(heatmap[joint,:,:]),pt= [uvd[joint,0],uvd[joint,1]], sigma=sigma)
    return heatmap

def get_max_preds( batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, width, width])
        '''
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        #pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        #pred_mask = pred_mask.astype(np.float32)

        #preds *= pred_mask
        return preds



def GetValuesX(dimension=64,num_channel=14):
    n=dimension
    num_channel=14
    vec=np.linspace(0,dimension-1,dimension).reshape(1,-1)
    Xs=np.linspace(0,dimension-1,dimension).reshape(1,-1)
    for i in range(n-1):
        Xs=np.concatenate([Xs,vec],axis=1)

    #Xs=np.repeat(Xs,num_channel,axis=0)
    Xs=np.expand_dims(Xs,axis=0)
    return nn.Parameter(torch.from_numpy(Xs))

def GetValuesY(dimension=64,num_channel=14):
    res=np.zeros((1,dimension*dimension))
    for i in range(dimension):
        res[0,(i*dimension):((i+1)*dimension)]=i
    res=np.expand_dims(res,axis=0)
    return nn.Parameter(torch.from_numpy(res))

def GetRandomWindow(img,win_size=31):
#img is a np.array of shape (w,h)
#The return window will be a random crop window of size (win_size+1,win_size+1) centered at (u,v)
    n=win_size
    length=np.int16(np.floor(n/2))
    window=np.zeros((n+1,n+1))
    u=np.random.randint(length,img.shape[0]-length-2)
    v=np.random.randint(length,img.shape[1]-length-2)
    window=img[(u-length):(u+length+2),(v-length):(v+length+2)]
    return window,u,v


def convert_xyz_to_uvd( xyz ):
    # both xyz and the resturned uvd will be np.array of size(num_joints,3)
    halfResX = 640/2;
    halfResY = 480/2;
    coeffX = 588.036865;
    coeffY = 587.075073;

    uvd = np.zeros(xyz.shape);
    uvd[:,0] = coeffX * xyz[:,0] / xyz[:,2] + halfResX
    uvd[:,1] = halfResY - coeffY * xyz[:,1] / xyz[:,2]
    uvd[:,2] = xyz[:,2]
    return uvd

def convert_uvd_to_xyz( uvd ):
    # both xyz and the resturned uvd will be np.array of size(num_joints,3)
    xRes = 640;
    yRes = 480;
    xzFactor = 1.08836710;
    yzFactor = 0.817612648;

    normalizedX = np.double(uvd[:,0]) / xRes - 0.5;
    normalizedY = 0.5 - np.double(uvd[:,1]) / yRes;

    xyz = np.zeros(uvd.shape);
    xyz[:,2] = np.double(uvd[:,2]);
    xyz[:,0] = normalizedX * xyz[:,2] * xzFactor;
    xyz[:,1] = normalizedY * xyz[:,2] * yzFactor;
    return xyz

def convert_depthImage_to_unichannel(img):
    # this function converts a depth map stored in 3-channel format to an image with one-channel
    return np.uint16(img[:,:,1])*256 + np.uint16(img[:,:,2])




def transformPose(x,transform_matrix):
    # transform the pose x of shape (14,3) using the matrix
    res=np.zeros(x.shape)
    for joint in range(x.shape[0]):
        t=transformPoint2D(x[joint], transform_matrix)
        res[joint, 0] = t[0]
        res[joint, 1] = t[1]
        res[joint, 2] = x[joint, 2]
    return res

def get_limb_heatmaps(input_uvd,parts,sigma=1,width=5,heatmap_size=(128,128)):
    #input_uvd is a tensor or np.array of size(n,2)
    # parts is a list of this format[[0,1],[4,8]] , e.g. means the first part limb start and end indicies are 0 and 1 respectively
    # returns an np.array of size(num_parts,heatmap_size[0],heatmap_size[1])
    
    h,w=heatmap_size
    num_parts=len(parts)
    heatmap = np.zeros((num_parts,h, w))
    mask = np.zeros((num_parts,h, w))
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    
    for i in range(num_parts):
        x1, y1 = input_uvd[parts[i][0]]    
        x2, y2 = input_uvd[parts[i][1]]  

        vector_length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # # normalize the vector to the unit size
        v1 = (x2 - x1)/(vector_length + 1e-10)  
        v2 = (y2 - y1)/(vector_length + 1e-10)

        dist_along_part = v1 * (x - x1) + v2 * (y - y1)
        dist_perpendicular_part = np.abs(v2 * (x - x1) + (-v1) * (y - y1))

        mask1 = dist_along_part >= 0
        mask2 = dist_along_part <= vector_length
        mask3 = dist_perpendicular_part <= width
        mask[i,...] = mask1 & mask2 & mask3

        heatmap[i,...]=mask[i]*np.exp(-dist_perpendicular_part / 2.0 / (sigma**2))#
        
    return heatmap,mask

def get_fingers(heatmpas,groups=[[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19], [20,21,22,23] ]):
    # heatmaps is np.array of size(num_parts,height,width)
    output=np.zeros((len(groups),heatmpas.shape[1],heatmpas.shape[2]))
    for g in range(len(groups)):
        output[g,...]=np.max(heatmpas[groups[g]],axis=0)
    return output


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    from torchvision.utils import make_grid
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
    
def compute_entropy(x):
    # x is a tensor of size (b,k,dim1,dim2)
    # each map of size (dim1,dim2) should be a probability map (their value sum up to one)
    
    x=x.view(x.shape[0],x.shape[1],-1)
    logs=torch.log(x)
    return torch.sum(-1*x*logs,dim=-1)


