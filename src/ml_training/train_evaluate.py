import logging

from src.utils.argumentlib import args
import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from src.utils.utils import rand_set_channels_to_zero


def local_train(model, optimizer, dataloader, epoch_number, device, dataset_name,
                img_index, label_index, channels, channel_map, cropped_input_size,
                total_modalities, loss_fun, scheduler):
    model.train()
    iterations_loss = []

    if args.client_number_iterations > 0:
        # Create a generator in order to not iterate over the whole dataset
        generator = iter(dataloader)
        for _ in tqdm(range(args.client_number_iterations), leave=True, position=0):
            try:
                data_batch = next(generator)
            except:
                print('[INFO] created a new iterator')
                # If generator is exhausted, create a new one
                generator = iter(dataloader)
                data_batch = next(generator)
            # Do a forward pass
            out, label = forward_pass_specific_model(model=model, batch=data_batch, device=device,
                                                     dataset_name=dataset_name,
                                                     img_index=img_index, label_index=label_index, channels=channels,
                                                     channel_map=channel_map, cropped_input_size=cropped_input_size,
                                                     total_modalities=total_modalities)
            # Backward pass
            optimizer.zero_grad()
            loss = loss_fun(out, label)
            loss.backward()
            optimizer.step()
            iterations_loss.append(loss.item())
        if scheduler is not None:
            logging.info('Scheduler step')
            scheduler.step()
    else:
        assert args.client_number_iterations != 0
        print('Training on whole dataset')
        # Iterate over the whole dataset
        logging.info(f'Training for {args.client_number_iterations} epochs')
        for _ in range(abs(args.client_number_iterations)):
            for _, data_batch in enumerate(tqdm(dataloader, leave=True, position=0)):
                # Do a forward pass
                out, label = forward_pass_specific_model(model=model, batch=data_batch, device=device,
                                                         dataset_name=dataset_name,
                                                         img_index=img_index, label_index=label_index, channels=channels,
                                                         channel_map=channel_map, cropped_input_size=cropped_input_size,
                                                         total_modalities=total_modalities)
                # Backward pass
                optimizer.zero_grad()

                loss = loss_fun(out, label)
                loss.backward()
                optimizer.step()
                iterations_loss.append(loss.item())
            if scheduler is not None:
                logging.info('Scheduler step')
                scheduler.step()

    return np.array(iterations_loss).mean()


def local_evaluate(model, dice_metric, dataloader, dataset_name, total_modalities, cropped_input_size,
                   post_trans, channel_map, device, modalities_present_during_training):
    model.eval()
    with torch.no_grad():
        seg_channel = 0
        val_images = None
        val_labels = None
        val_outputs = None
        dice_metric.reset()

        # For the validation loss
        for i, data in enumerate(dataloader):

            input_data = torch.from_numpy(np.zeros((1, len(total_modalities), data[0].shape[2],
                                                    data[0].shape[3], data[0].shape[4]),
                                                   dtype=np.float32))
            input_data[:, channel_map, :, :, :] = data[0][:, modalities_present_during_training, :, :, :]
            input_data = input_data.to(device)
            if dataset_name == "BRATS" and args.BRATS_two_channel_seg:
                label = data[1][:, [0], :, :, :].to(device)
            else:
                label = data[1].to(device)

            roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
            sw_batch_size = 1
            val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=label)

        dice_score = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

        return dice_score


def forward_pass_specific_model(model, batch, device, dataset_name, img_index, label_index,
                                channels, channel_map, cropped_input_size, total_modalities):

    if dataset_name == "BRATS":
        if args.randomly_drop:
            modalities_remaining, batch[img_index] = rand_set_channels_to_zero(batch[img_index])
            # Shape of the new tensor
            new_shape = (batch[label_index].size(0), 1, batch[label_index].size(2),
                         batch[label_index].size(3), batch[label_index].size(4))
            # Create a new tensor with the desired shape
            label = torch.zeros(new_shape).to(device)
            for sample_idx in range(batch[img_index].shape[0]):
                # this part is only relevant for BRATS when doing 2 channel segmentation
                if (0 not in modalities_remaining[sample_idx]) and (3 not in modalities_remaining[sample_idx]):
                    # Edema cannot be seen so change segmentation to labels without edema
                    seg_channel = 1
                else:
                    seg_channel = 0
                if args.BRATS_two_channel_seg:
                    label[sample_idx, 0, :, :, :] = batch[label_index][sample_idx, [seg_channel], :, :, :]
                else:
                    label[sample_idx] = batch[label_index][sample_idx]

        else:
            label = batch[label_index].to(device)

        input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0], len(total_modalities),
                                                cropped_input_size[0], cropped_input_size[1],
                                                cropped_input_size[2]), dtype=np.float32))
        input_data[:, channel_map, :, :, :] = batch[img_index]
        input_data = input_data.to(device)
    else:
        if args.randomly_drop:
            _, batch[img_index] = rand_set_channels_to_zero(batch[img_index])  # ATLAS WILL ALWAYS BE ONE
        input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0], len(total_modalities),
                                                cropped_input_size[0], cropped_input_size[1],
                                                cropped_input_size[2]), dtype=np.float32))
        input_data[:, channel_map, :, :, :] = batch[img_index]
        input_data = input_data.to(device)

        label = batch[label_index].to(device)

    out = model(input_data)

    return out, label
