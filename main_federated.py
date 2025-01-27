import torch
import os
import sys
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from torch.backends import cudnn

from src.dataloaders.config import Config
from src.dataloaders.dataloaders import create_dataloader
from src.networks.theory_UNET import theory_UNET
import numpy as np
import random
from src.federated.server import Server
from src.federated.client import Client
import wandb
import copy

from src.utils.argumentlib import args
from pathlib import Path
import logging
from src.ml_training.losses import get_loss
from src.utils.utils import map_channels

os.chdir(sys.path[0])

if __name__ == "__main__":

    wandb_metrics = {}

    if args.wandb_projectname is not None:
        run_id = None
        if args.save_checkpoint:
            # Create ID
            run_id = wandb.util.generate_id()
        if args.resume_training:
            # Load ID from checkpoint
            run_id_save_path = Path('./output') / args.experiment_name / 'checkpoint' / f'global.pth'
            checkpoint_run_id = torch.load(run_id_save_path)
            run_id = checkpoint_run_id['run_id']
            wandb.init(project=args.wandb_projectname, id=run_id, name=args.experiment_name, config=args, resume='must')
        else:
            wandb.init(project=args.wandb_projectname, name=args.experiment_name, config=args, id=run_id)

    # Load config
    data_config = Config(config_file_name=args.config_file, brats_flag=args.BRATS_two_channel_seg)
    # DATA path
    data_path = Path(data_config.root_path)
    output_path = Path('./output') / args.experiment_name



    logging.basicConfig(filename=os.path.join(output_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    # Also add to StdOut
    logFormatter = logging.Formatter('[%(asctime)s] %(message)s')
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        logging.info(f"Setting seed: {args.seed} [NOTE] We don't have full determinism because of non-deterministic 3D PyTorch implementation")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logging.info(f"CUDA is available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available. Please make sure you have a compatible "
                        "GPU and have installed the necessary CUDA drivers.")

    img_index = 0
    label_index = 1
    # NORMALIZATION
    if args.norm == "GROUP":
        args.norm = ("group", {"num_groups": 16})
        logging.info('Norm: ', args.norm)

    # parameter for data
    logging.info("Learning Rate: ", args.lr)
    logging.info("Number of Workers: ", args.num_workers)
    logging.info("Batch size: ", args.batch_size)
    logging.info("Random Drop Modalities:", args.randomly_drop)

    logging.info("Running on GPU:" + str(args.gpu))
    logging.info("Running for epochs:" + str(args.E))

    cuda_id = "cuda:" + str(args.gpu)
    # Create the clients
    all_client_names = args.evaluation_datasets
    # Sanity check Verify whether it is a subset
    assert set(args.datasets) <= set(all_client_names)

    if int(args.gpu) == -1:
        device = torch.device("cuda")
    else:
        device = torch.device(cuda_id)

    # Loop for all the size and modalities
    datasetlist = args.datasets
    total_modalities = []
    total_modalities = set(total_modalities)
    for dataset in datasetlist:
        total_modalities = total_modalities.union(set(data_config.datasets[dataset]['channels']))

    total_modalities = sorted(list(total_modalities))

    logging.info("Total modalities: ", total_modalities)

    # Loop for allocate channel
    channel_map = {}
    for dataset in all_client_names:
        channel_map[dataset] = map_channels(data_config.datasets[dataset]['channels'], total_modalities)
        logging.info("channel map:", dataset, channel_map[dataset])

    train_loaders = {}
    val_loaders = {}

    # get dataloader
    for dataset in all_client_names:
        logging.info("Training: ", dataset)
        val_size = data_config.datasets[dataset]['total_size'] - data_config.datasets[dataset]['train_size']
        logging.info(f'Validation set size for dataset {dataset} is {val_size}')

        train_loaders[dataset], val_loaders[dataset] = create_dataloader(data_config=data_config,
                                                                         dataset_name=dataset,
                                                                         train_n_samples=data_config.datasets[dataset]['train_size'],
                                                                         num_workers=args.num_workers, batch_size=args.batch_size)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    logging.info("in_channels=", len(total_modalities))
    logging.info("batch size = ", args.batch_size)

    # configuration for different network architectures
    net_glob = theory_UNET(in_channels=len(total_modalities), norm=args.norm).to(device)

    # Create the server
    server = Server(global_model=net_glob, client_names_training_enabled=args.datasets,
                    aggregation_method=args.aggregation_method, device=device)

    for client_name in all_client_names:
        # Unpack the tuple returned by parse_loss_argument
        loss_name, loss_args = args.loss_argument
        logging.info(f"Using loss function {loss_name} with arguments {loss_args}")

        loss_function = get_loss(loss_name, **loss_args)
        # Optimizer parameters and loss function
        if args.optimizer == "ADAM":
            optimizer_class = torch.optim.Adam
            optimizer_params = {"lr": args.lr}
        elif args.optimizer == "SGD":
            optimizer_class = torch.optim.SGD
            optimizer_params = {"lr": args.lr}
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

        modalities_present_during_training = [data_config.datasets[client_name]['channels'].index(element)
                                              for element in data_config.datasets[client_name]['channels'] if element in total_modalities]
        # Sanity check
        logging.info(f"Client {client_name} has modalities {modalities_present_during_training} during training")
        if client_name in args.datasets:
            assert len(modalities_present_during_training) == len(data_config.datasets[client_name]['channels'])

        client = Client(model=copy.deepcopy(net_glob), optimizer_class=optimizer_class,
                        optimizer_params=optimizer_params, training_dataset_length=data_config.datasets[client_name]['train_size'],
                        name=client_name, device=device, train_dataloader=train_loaders[client_name],
                        val_dataloader=val_loaders[client_name], server=server, criterion=loss_function,
                        channels=data_config.datasets[client_name]['channels'], channel_mapping=channel_map[client_name],
                        modalities_present_during_training=modalities_present_during_training)

        server.add_client(client)

    model_save_path = output_path / 'models'

    wandb_metrics = {}

    start_epoch = 0
    # Resume training
    if args.resume_training:
        start_epoch = server.load_checkpoint()

    for epoch in range(start_epoch, args.E):
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{args.E}")
        # Server starts training the clients
        server.train(wandb_metrics=wandb_metrics, epoch_number=epoch+1,
                     img_index=img_index, label_index=label_index, cropped_input_size=data_config.cropped_input_size,
                     total_modalities=total_modalities)

        # ------------------ VALIDATION ------------------------
        if (epoch + 1) % args.validation_interval == 0:
            server.evaluate(dice_metric=dice_metric, total_modalities=total_modalities, cropped_input_size=data_config.cropped_input_size,
                            pos_trans=post_trans, step=epoch + 1, wandb_metrics=wandb_metrics, save_latest=True)
            wandb.log(wandb_metrics, step=epoch + 1)

        # Save checkpoint
        if args.save_checkpoint:
            server.save_checkpoint(step=epoch + 1)

    wandb.finish()