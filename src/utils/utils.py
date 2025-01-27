import numpy as np
import torch

def rand_set_channels_to_zero(batch_data: torch.Tensor):
    modified_batch = batch_data.clone()
    modalities_remaining = []
    for sample_idx in range(batch_data.shape[0]):
        num_channels = batch_data.shape[1]

        # Skip dropout if there is only one channel
        if num_channels > 1:
            # Randomly choose number of modalities to drop
            number_of_dropped_modalities = torch.randint(0, num_channels, (1,))

            # Randomly choose modalities to set to 0
            modalities_dropped = torch.randperm(num_channels)[:number_of_dropped_modalities]

            # Set the selected channels to 0 for the current sample
            modified_batch[sample_idx, modalities_dropped, :, :, :] = 0.
            modalities_remaining.append(list(set(np.arange(num_channels)) - set(modalities_dropped)))

    return modalities_remaining, modified_batch

def map_channels(dataset_channels, total_modalities):
    channel_map = []
    for channel in dataset_channels:
        for index, modality in enumerate(total_modalities):
            if channel == modality:
                channel_map.append(index)
    return channel_map