import torch
import copy
import logging

from src.utils.network_utils import get_normalization_layer_names


def fed_avg(server, clients, equal_weighting=False):
    logging.info('Performing FedAvg')
    with torch.no_grad():
        # Create a copy of the model to get the architecture
        w_avg = copy.deepcopy(server.global_model_weights)
        total_datapoints = sum(client.training_dataset_length for _, client in clients.items())

        for k in server.global_model_weights.keys():
            if equal_weighting:
                w_avg[k] = sum(c.model_weights[k] for _, c in clients.items()) / len(clients)
            else:
                # First set to zero in order to compute the weighted average
                w_avg[k] = torch.zeros_like(w_avg[k])
                for _, c in clients.items():
                    if 'num_batches_tracked' in k:
                        # Num_batches_tracked is a single scalar that increments by 1 every time forward is
                        # called by _BatchNorm
                        # However, it is only used if momentum is set to None (which is not the default value)
                        # Our methods do use momentum, so we can ignore this parameter and all our models are trained
                        # using same number of batches
                        w_avg[k] = c.model_weights[k]
                    else:
                        w_avg[k] += (c.training_dataset_length / total_datapoints) * c.model_weights[k]

        server.update_model(new_model=w_avg)
        return w_avg


def fed_bn(server, clients, equal_weighting=False):
    logging.info('Performing FedBN')
    # First run fed_avg. This also updates the server model
    w_avg = fed_avg(server=server, clients=clients, equal_weighting=equal_weighting)
    # Create a copy to be able to remove certain keys
    w_avg_non_norm_params = copy.deepcopy(w_avg)

    norm_layer_names = get_normalization_layer_names(state_dict=w_avg_non_norm_params, model=server.global_model)
    for key in norm_layer_names:
        del w_avg_non_norm_params[key]
    return w_avg, w_avg_non_norm_params
