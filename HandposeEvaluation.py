"""
Created on Fri Nov  1 11:25:13 2019

@author: mrezaei
"""
import numpy as np
import os

class HandposeEvaluation(object):
 
    def __init__(self, predictions, gt):
        # both predictions and gt should be np.array of shape (batch,num_key,3)
        self.gt=gt
        self.joints=predictions
    def getMeanError(self):
        """
        get average error over all joints, averaged over sequence
        :return: mean error
        """
        return np.nanmean(np.nanmean(np.sqrt(np.square(self.gt - self.joints).sum(axis=2)), axis=1))
    
    def GetMeanErrorPerJoint(self):
        d=np.abs(self.gt-self.joints)
        distances=np.linalg.norm(d,axis=2)
        return np.mean(distances,axis=0)
    
    def getNumFramesWithinMaxDist(self, dist):
        """
        calculate the number of frames where the maximum difference of a joint is within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (np.nanmax(np.sqrt(np.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def getErrorPerDimension(self):
        d=np.abs(self.gt-self.joints)
        xmean=np.mean(d[:,:,0])
        ymean=np.mean(d[:,:,1])
        zmean=np.mean(d[:,:,2])
        print("mean along X direction = ",xmean);
        print("mean along Y direction = ",ymean);
        print("mean along Z direction = ",zmean);
        return;
