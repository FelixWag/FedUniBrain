import torch.nn as nn
import torchvision
from src.networks.theory_UNET import theory_UNET
from src.utils.argumentlib import args


""" Linear increase and then decrease scheduler """
warmup_epochs = 50
# Create a learning rate scheduler with a warm-up phase based on epochs
def lr_lambda(current_epoch):
    if current_epoch < 50:
        return (float(current_epoch)+1) / float(max(1, 50))
    elif 50 <= current_epoch <= 150:
        return 1.0
    else:
        return max(0.0, 1.0 - (current_epoch - 150) / float(max(1, args.E-150)))
        #return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_epoch - warmup_epochs) / max(1, args.E - warmup_epochs))))

def get_normalization_layer_names(state_dict, model):
    # For each new architecture we need to specify the corresponding layer names
    if isinstance(model, torchvision.models.densenet.DenseNet):
        return [k for k in list(state_dict.keys()) if 'norm' in k]
    elif isinstance(model, theory_UNET):
        return [k for k in list(state_dict.keys()) if '.N.' in k]
    else:
        raise ValueError(f"{type(model)} is not supported")


def reset_bn_running_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
            m.reset_running_stats()


def set_bn_to_train_disable_grad(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
            m.train()
            m.requires_grad_(False)


def estimate_bn_stats(model, dataloader, device, n_iter=2):
    if n_iter == 0:
        # If we don't want to estimate the statistics, just return
        return
    print("Estimating fedBN statistics on unseen client")
    # Set model to eval
    model.eval()
    # Reset BN params
    reset_bn_running_stats(model)
    # set BN params to train + learnables requires to false
    set_bn_to_train_disable_grad(model)
    # Iterate over model k times
    for i in range(n_iter):
        for data, _ in dataloader:
            data = data.to(device)
            model(data)
    # Enable grad for the whole model again
    model.requires_grad_(True)
    # Set model back to train
    model.train()
