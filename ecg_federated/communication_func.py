import torch
import numpy as np
from copy import deepcopy

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.algorithm == 'fedbn' or args.algorithm == "fedpxn":
            for key in server_model.state_dict().keys():
                if 'norm' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)

                    for client_idx in range(len(args.data_list)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]

                    server_model.state_dict()[key].data.copy_(temp)

                    for client_idx in range(len(args.data_list)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        else: # fedavg, fedprox
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key :
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    
                    for client_idx in range(len(args.data_list)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]

                    server_model.state_dict()[key].data.copy_(temp)
                    
                    for client_idx in range(len(args.data_list)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models

################################################################################################

def communication_fedopt(args, server_model, models, client_weights, m_t, v_t):
    with torch.no_grad():
        # 1. Make pseudo gradient
        pseudo_gradient = deepcopy(server_model)
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                # server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                pseudo_gradient.state_dict()[key].data.copy_(models[0].state_dict()[key])

            else:
                temp = torch.zeros_like(server_model.state_dict()[key])

                for client_idx in range(len(args.data_list)):
                    client_temp = models[client_idx].state_dict()[key] - server_model.state_dict()[key]
                    temp += client_temp * client_weights[client_idx]

                pseudo_gradient.state_dict()[key].data.copy_(temp)

        # 2. Update by each optimizer

        # m_t (momentum calculation)
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                m_t.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = args.beta_1 * m_t.state_dict()[key] + (1 - args.beta_1) * pseudo_gradient.state_dict()[key]
                m_t.state_dict()[key].data.copy_(temp)

        # v_t (second moment calculation)
        if args.algorithm == "fedadam" :
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    v_t.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = args.beta_2 * v_t.state_dict()[key] + (1 - args.beta_2) * torch.pow( pseudo_gradient.state_dict()[key], 2)
                    v_t.state_dict()[key].data.copy_(temp)

        elif args.algorithm == "fedadagrad" :
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    v_t.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = v_t.state_dict()[key] + torch.pow( pseudo_gradient.state_dict()[key], 2)
                    v_t.state_dict()[key].data.copy_(temp)

        else : # fedyogi
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    v_t.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = v_t.state_dict()[key] - (1 - args.beta_2) * torch.pow( pseudo_gradient.state_dict()[key], 2) * torch.sign(v_t.state_dict()[key] - torch.pow( pseudo_gradient.state_dict()[key], 2))
                    v_t.state_dict()[key].data.copy_(temp)


        # 3. update global model and local models
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = server_model.state_dict()[key] + args.server_learning_rate * ( m_t.state_dict()[key] / (  torch.sqrt(v_t.state_dict()[key]) + args.tau ) )
                server_model.state_dict()[key].data.copy_(temp)
                
                for client_idx in range(len(args.data_list)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models, m_t, v_t

#############################################################

def communication_FedDyn(args, server_model, models, client_weights, cld_model, prev_grads):
                
    with torch.no_grad():
        # aggregate params        
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:                
                sum_theta = torch.zeros_like(server_model.state_dict()[key])
                grad_theta = torch.zeros_like(server_model.state_dict()[key])

                for client_idx in range(len(args.data_list)):
                    sum_theta += models[client_idx].state_dict()[key]
                    grad_theta += prev_grads[client_idx].state_dict()[key]
                
                cld_model.state_dict()[key].data.copy_( (1./len(args.data_list)) * (sum_theta + grad_theta) ) 

                server_model.state_dict()[key].data.copy_( (1./len(args.data_list)) * sum_theta )
    
                for client_idx in range(len(args.data_list)):
                    models[client_idx].state_dict()[key].data.copy_(cld_model.state_dict()[key])
        #####

    return server_model, models, cld_model