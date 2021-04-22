import torch

########## Inspection ######################

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


def count_parameters(model):
    # counts the number of learned parameter in the model (in Millions)
    # the model is of type nn.Module
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


########################### Utils ############################


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    #alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
