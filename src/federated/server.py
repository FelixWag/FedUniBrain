import statistics

import torch
import torch.nn as nn

from src.federated.client import Client
from src.federated.federated_algorithms import fed_avg, fed_bn
from src.utils.argumentlib import args

from pathlib import Path

from src.utils.logging_utils import update_wandb_metrics
import wandb


class Server:
    def __init__(self, global_model: torch.nn.Module, client_names_training_enabled: list[str],
                 aggregation_method: str, device):
        self._global_model = global_model
        self.all_clients = {}
        self.training_clients = {}
        self._client_names_training_enabled = client_names_training_enabled
        self._aggregation_method = aggregation_method
        self._best_selection_metric = 0.0

        self._device = device
        # Move both models to device
        self._global_model.to(device)

    @property
    def global_model(self) -> nn.Module:
        return self._global_model

    @property
    def global_model_weights(self) -> dict:
        return self._global_model.state_dict()

    def update_model(self, new_model, strict=True) -> None:
        if isinstance(new_model, nn.Module):
            self._global_model.load_state_dict(new_model.state_dict(), strict=strict)
        elif isinstance(new_model, dict):
            self._global_model.load_state_dict(new_model, strict=strict)
        else:
            raise ValueError('Invalid input: new_model must be an instance of nn.Module or a state dictionary.')

    def add_client(self, client: Client) -> None:
        assert client.name not in self.all_clients
        self.all_clients[client.name] = client
        if client.name in self._client_names_training_enabled:
            self.training_clients[client.name] = client

    def train(self, wandb_metrics: dict, epoch_number, img_index, label_index,
              cropped_input_size, total_modalities) -> None:
        losses = []
        # First, local training on clients
        for _, client in self.training_clients.items():
            loss = client.train(wandb_metrics=wandb_metrics, epoch_number=epoch_number, img_index=img_index,
                                label_index=label_index, cropped_input_size=cropped_input_size,
                                total_modalities=total_modalities, method=args.training_algorithm)
            losses.append(loss)
        print(f'Average loss on clients: {statistics.fmean(losses):.2f}')
        # Aggregate the client model and update global model
        with torch.no_grad():
            if self._aggregation_method == 'fedavg':
                w_avg = fed_avg(server=self, clients=self.training_clients, equal_weighting=args.equal_weighting)
                # Update/Send the client models
                for _, client in self.all_clients.items():
                    client.update_model(w_avg)
            elif self._aggregation_method == 'fedbn':
                assert args.norm == 'BATCH'
                w_avg, w_avg_non_norm_params = fed_bn(server=self, clients=self.training_clients,
                                                      equal_weighting=args.equal_weighting)
                for _, client in self.all_clients.items():
                    # If it is in training, then update with non_norm_params, to keep them local
                    if client.name in self._client_names_training_enabled:
                        # We need to set strict to false since we are not loading the normalization layers
                        client.update_model(w_avg_non_norm_params, strict=False)
                    else:
                        # Update rest of the models with the server model
                        client.update_model(w_avg)
                        if args.estimate_bn_stats:
                            client.estimate_bn_stats('train')
            else:
                raise ValueError(f"Unsupported aggregation method: {self._aggregation_method}")

    def evaluate(self, dice_metric, total_modalities, cropped_input_size, pos_trans, step,
                 wandb_metrics, save_best_model=True, save_latest=False) -> None:

        client_dice_scores = []

        for _, client in self.all_clients.items():
            dice_score = client.evaluate(dice_metric=dice_metric,
                                         total_modalities=total_modalities,
                                         cropped_input_size=cropped_input_size,
                                         pos_trans=pos_trans, step=step,
                                         save_best_model=save_best_model,
                                         wandb_metrics=wandb_metrics,
                                         save_latest=save_latest)

            client_dice_scores.append(dice_score)

        # Compute mean loss and accuracy
        mean_dice_score = statistics.fmean(client_dice_scores)

        # Log to console and wandb
        print(
            f'Evaluate server on step {step}. Mean dice score is: {mean_dice_score:.2%}')
        if args.wandb_projectname is not None:
            update_wandb_metrics(wandb_metrics=wandb_metrics, train_val_test='val', mean_dice=mean_dice_score)

        if save_best_model:
            if mean_dice_score > self._best_selection_metric:
                self._best_selection_metric = mean_dice_score
                self.save_model(which_model='dice', step=step)
        if save_latest:
            self.save_model(which_model='latest', step=step)

    def save_model(self, which_model, step) -> None:
        assert which_model in ['dice', 'latest']
        print(f'Improvement achieved: Saving global model on step {step}')
        save_path = Path('./output') / args.experiment_name / 'models' / f'global_{which_model}.pth'
        torch.save(self.global_model.state_dict(), save_path)
        print(f'Saved global model successfully in {save_path}')

    def load_global_model(self, model_directory_path: Path, which_model: str) -> None:
        assert which_model in ['dice', 'latest']
        print(f'Loading global model from {model_directory_path}')
        load_path = model_directory_path / f'global_{which_model}.pth'
        self.update_model(torch.load(load_path, map_location=self._device))
        print(
            f'Loaded global model successfully from {model_directory_path / f"global_{which_model}.pth"}')

    def load_best_global_model(self, evaluation_metric_name) -> None:
        save_path = Path('./output') / args.experiment_name / 'models'
        print(f'Loading best global model')
        self.load_global_model(model_directory_path=save_path, which_model=evaluation_metric_name)

    def load_best_global_and_client_models(self, evaluation_metric_name) -> None:
        self.load_best_global_model(evaluation_metric_name=evaluation_metric_name)
        for _, client in self.all_clients.items():
            client.load_best_model(evaluation_metric_name=evaluation_metric_name)

    def load_global_and_client_models(self, model_directory_path: Path, which_model: str) -> None:
        self.load_global_model(model_directory_path=model_directory_path, which_model=which_model)
        for _, client in self.all_clients.items():
            client.load_model(path=model_directory_path, which_model=which_model)

    def save_checkpoint(self, step):
        print(f'Checkpoint: Saving checkpoint for global server on step {step}')
        save_path = Path('./output') / args.experiment_name / 'checkpoint' / f'global.pth'
        checkpoint = {
            'epoch': step,
            'model_state_dict': self._global_model.state_dict(),
            'run_id': wandb.run.id
        }
        torch.save(checkpoint, save_path)
        for _, client in self.all_clients.items():
            client.save_checkpoint(step)

    def load_checkpoint(self):
        save_path = Path('./output') / args.experiment_name / 'checkpoint' / f'global.pth'
        checkpoint = torch.load(save_path, map_location=self._device)
        epoch = checkpoint['epoch']
        self.update_model(checkpoint['model_state_dict'])
        for _, client in self.all_clients.items():
            client.load_checkpoint()
        print(f'Loaded checkpoint from global model successfully from {save_path}')
        return epoch
