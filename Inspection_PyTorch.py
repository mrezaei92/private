import torch


def compare(state_dic1,state_dic2):
    # both state_dic1 and state_dic2 are of type collections.OrderedDict
    # This function compares the dic_state of models (should be the same), and checks for the tensor that are
    # unchanged across the models. Example Usage: m1=torch.load(mod1),m2=torch.load(mod1),compare(m1,m2)
    keys=[];c=0
    for key in state_dic1.keys():
        if torch.sum(state_dic1[key]==state_dic2[key])==state_dic1[key].numel():
            c=c+1
            keys.append(key)
            
    print("Num tensors unchanged: ", c)
    return keys
