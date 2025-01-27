from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.ml_training.train_evaluate import local_evaluate, local_train
from src.utils.logging_utils import update_wandb_metrics_client
from src.utils.argumentlib import args

from src.utils.network_utils import estimate_bn_stats, lr_lambda


class Client:
    def __init__(self, model: torch.nn.Module, optimizer_class: type, optimizer_params: dict, training_dataset_length: int,
                 name: str, device, train_dataloader, val_dataloader, server, criterion, channels, channel_mapping,
                 modalities_present_during_training: list, scheduler_class: type = None, scheduler_params: dict = None):
        self._model = model
        self._optimizer = optimizer_class(self._model.parameters(), **optimizer_params)
        self._training_dataset_length = training_dataset_length
        self._scheduler = None
        self._name = name
        self._device = device
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

        if scheduler_class is not None:
            self._scheduler = scheduler_class(self._optimizer, **scheduler_params)
        if args.scheduler:
            assert not (args.scheduler and args.drop_learning_rate)
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self._server = server
        self._criterion = criterion
        # THESE are the available MRI channels
        self._channels = channels
        self._channel_mapping = channel_mapping
        self._modalities_present_during_training = modalities_present_during_training

        self.best_selection_metric = -1.0

        # Move model to device
        self._model.to(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    @property
    def training_dataset_length(self) -> int:
        return self._training_dataset_length

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def model_weights(self) -> dict:
        return self._model.state_dict()

    @property
    def name(self) -> str:
        return self._name

    def get_dataloader(self, train_val: str):
        if train_val == 'train':
            return self._train_dataloader
        elif train_val == 'val':
            return self._val_dataloader
        else:
            raise ValueError(
                f'Invalid input: train_test_val must be either train, val or test but was {train_val}.')

    def update_model(self, new_model, strict=True) -> None:
        if isinstance(new_model, nn.Module):
            self._model.load_state_dict(new_model.state_dict(), strict=strict)
        elif isinstance(new_model, dict):
            self._model.load_state_dict(new_model, strict=strict)
        else:
            raise ValueError('Invalid input: new_model must be an instance of nn.Module or a state dictionary.')

    def estimate_bn_stats(self, train_test_val) -> None:
        estimate_bn_stats(model=self._model, dataloader=self.get_dataloader(train_val=train_test_val),
                          device=self._device)


    def train(self, wandb_metrics: dict, epoch_number, img_index, label_index, cropped_input_size,
              total_modalities, method='default'):
        print(f'Begin local training on client: {self.name}')
        if method == 'default':
            loss = local_train(model=self._model, optimizer=self._optimizer, dataloader=self.get_dataloader('train'),
                               epoch_number=epoch_number, device=self._device, dataset_name=self.name, img_index=img_index,
                               label_index=label_index, channels=self._channels, channel_map=self._channel_mapping,
                               cropped_input_size=cropped_input_size, total_modalities=total_modalities,
                               loss_fun=self._criterion, scheduler=self._scheduler)
        else:
            raise ValueError(f'Invalid method: {method}')

        update_wandb_metrics_client(wandb_metrics=wandb_metrics, train_val_test='train', client_name=self.name,
                                    loss=loss, lr=self._optimizer.param_groups[0]['lr'])

        print(f'Client {self.name} - Training loss: {loss:.2}' + '\n' + '-' * 50)
        return loss

    def evaluate(self, dice_metric, total_modalities, cropped_input_size,
                 pos_trans, step, wandb_metrics, save_best_model=True, save_latest=False):
        print('-' * 50 + '\n' + f'Evaluate client: {self.name} on step {step}')

        dice_score = local_evaluate(model=self._model, dice_metric=dice_metric, dataloader=self.get_dataloader('val'),
                                    dataset_name=self._name, total_modalities=total_modalities,
                                    cropped_input_size=cropped_input_size, post_trans=pos_trans,
                                    channel_map=self._channel_mapping, device=self._device,
                                    modalities_present_during_training=self._modalities_present_during_training)

        if save_best_model:
            if dice_score > self.best_selection_metric:
                self.best_selection_metric = dice_score
                self.save_model(which_model='dice', step=step)
        if save_latest:
            self.save_model(which_model='latest', step=step)

        # Log to console and wandb
        print(f'Client {self.name} - Val dice score: {dice_score:.4f}' + '\n' + '-' * 50)
        if args.wandb_projectname is not None:
            update_wandb_metrics_client(wandb_metrics=wandb_metrics, train_val_test='val', client_name=self.name,
                                        mean_dice=dice_score)

        return dice_score

    def save_model(self, which_model, step):
        assert which_model in ['dice', 'latest']
        print(f'Improvement achieved: Saving model of client {self.name} on step {step}')
        save_path = Path('./output') / args.experiment_name / 'models' / f'{self.name}_{which_model}.pth'
        torch.save(self._model.state_dict(), save_path)
        print(f'Saved model of client {self.name} successfully in {save_path}')

    def load_model(self, path: Path, which_model: str) -> None:
        assert which_model in ['dice', 'latest']
        print(f'Loading global model from {path}')
        load_path = path / f'{self.name}_{which_model}.pth'
        self.update_model(torch.load(load_path, map_location=self._device))
        print(f'Loaded global model successfully from {path / f"{self.name}_{which_model}.pth"}')

    def load_best_model(self, evaluation_metric_name) -> None:
        save_path = Path('./output') / args.experiment_name / 'models'
        print(f'Loading best model of client {self.name} from {save_path}')
        self.load_model(path=save_path, which_model=evaluation_metric_name)

    def save_checkpoint(self, step):
        print(f'Checkpoint: Saving checkpoint of client {self.name} on step {step}')
        save_path = Path('./output') / args.experiment_name / 'checkpoint' / f'{self.name}.pth'
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        if self._scheduler is not None:
            checkpoint['scheduler'] = self._scheduler.state_dict()

        torch.save(checkpoint, save_path)

    def load_checkpoint(self):
        save_path = Path('./output') / args.experiment_name / 'checkpoint' / f'{self.name}.pth'
        checkpoint = torch.load(save_path, map_location=self._device)
        self.update_model(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        if self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'Loaded checkpoint of client {self.name} successfully from {save_path}')
