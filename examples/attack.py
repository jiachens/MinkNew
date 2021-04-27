'''
Description: 
Autor: Jiachen Sun
Date: 2021-03-07 16:29:42
LastEditors: Jiachen Sun
LastEditTime: 2021-04-16 22:25:05
'''

import torch
import numpy as np
import torch.nn.functional as F
import copy
from examples.classification_modelnet40 import create_input_batch

def pgd_attack(model,data,labels,flag,inisde,index,device,quantization_size,eps=0.05,alpha=0.01,iters=7,repeat=1):
    model.eval()
    max_loss = -1
    best_examples=None

    data['features'] = data['features'].cuda().to(device)

    if inisde:
        eps_lower = torch.floor(data['features'] / quantization_size) * quantization_size
        eps_upper = torch.ceil(data['features'] / quantization_size) * quantization_size
    
    for i in range(repeat):
        adv_data = {}
        adv_feature = data['features'].clone().cuda().to(device)
        adv_feature.detach()
        adv_feature = adv_feature + (torch.rand_like(adv_feature)*eps*2-eps)
        delta = adv_feature - data['features']
        delta = torch.clamp(delta,-eps,eps)
        adv_data['features'] = data['features']+delta
        if inisde:
            adv_data['features'] = torch.max(torch.min(adv_data['features'],eps_upper),eps_lower)
        # adv_data['features'] = adv_feature.cuda().to(device)
        adv_data['coordinates'] = data['coordinates'].clone()
        adv_data['coordinates'][:,1:] = adv_data['features']


        input_adv = create_input_batch(adv_data,flag,device,quantization_size)
        orig =  input_adv._C.clone()

        # adv_data.detach()
        for i in range(iters):
            input_adv._F.requires_grad=True
            logits = model(input_adv)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()

            if loss > max_loss:
                max_loss=loss
                best_examples=input_adv
            
            with torch.no_grad():
                adv_data['features'] = adv_data['features'] + alpha * input_adv._F.grad.sign()
                if not inisde:
                    if index:
                        delta = adv_data['features']- data['features']
                        delta = torch.clamp(delta,-eps,eps)
                        adv_data['features'] = data['features']+delta
                        adv_data['coordinates'][:,1:] = adv_data['features']
                        input_adv = create_input_batch(adv_data,flag,device,quantization_size)
                    else:
                        delta = adv_data['features']- data['features']
                        delta = torch.clamp(delta,-eps,eps)
                        adv_data['features'] = data['features']+delta
                        adv_data['coordinates'][:,1:] = adv_feature
                        input_adv = create_input_batch(adv_data,flag,device,quantization_size)
                else:
                    delta = adv_data['features']- data['features']
                    delta = torch.clamp(delta,-eps,eps)
                    adv_data['features'] = data['features']+delta
                    adv_data['features'] = torch.max(torch.min(adv_data['features'],eps_upper),eps_lower)
                    # delta = adv_data['features']- data['features']
                    # delta = torch.clamp(delta,-eps,eps)
                    # adv_data['features'] = data['features']+delta
                    adv_data['coordinates'][:,1:] = adv_data['features']
                    # print(torch.max(adv_data['features']),torch.min(adv_data['features']))
                    input_adv = create_input_batch(adv_data,flag,device,quantization_size)
                    # print(torch.max(input_adv._C - orig))
               #If points outside the unit cube are invalid then
                # adv_data = torch.clamp(adv_data,-1,1)

        logits = model(input_adv)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        
        if loss > max_loss:
            max_loss=loss
            best_examples=input_adv
    
    # print(input_adv._F, input_adv._C)
    return input_adv