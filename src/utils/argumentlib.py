import argparse
import sys
import os

from pathlib import Path

parser = argparse.ArgumentParser(description='Multi-task MRI Arguments')

os.chdir(sys.path[0])


def parse_args():
    print("[IMPORTANT] parse_args gets exectuded")
    parser.add_argument('--config_file', type=str, default='config_brain_mri.ini', help='Configuration filename for the datasets.')
    parser.add_argument('--gpu', type=str, default='1', help='Which GPU to use. If None, use CPU')
    parser.add_argument('--client_number_iterations', type=int, default=58, help='Number of iterations per client. If negative then every client does the absolute value number of epochs over the whole dataset')
    parser.add_argument('--E', type=int, default=300, help='Number of communication rounds')
    parser.add_argument('--experiment_name', type=str, default='exper0', help='Name of the experiment.')
    parser.add_argument('--deterministic', default=True, action=argparse.BooleanOptionalAction,
                        help='whether use deterministic training (Note: We cannot guarantee full determinism due to non-deterministic implementation of AvgPool3D)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--scheduler', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to use a scheduler')

    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'],
                        help='Specify the optimizer to use. Current choices:'
                             'SGD... Stochastic Gradient Descent'
                             'ADAM... ADAM optimizer. Note currently only SGD is supported for StarAlign')

    parser.add_argument('--datasets', nargs='+',
                        help='List of client names that are enabled for training.',
                        default=['ATLAS', 'BRATS'], type=str)

    parser.add_argument('--evaluation_datasets', nargs='+', default=['ATLAS', 'BRATS'],
                        help='List of client names that are enabled for evaluation.', type=str)

    parser.add_argument('--norm', type=str, default='INSTANCE', choices=['INSTANCE', 'BATCH', 'GROUP', "WSConv", "ScaledStdConv"],
                        help='Which normalisation to use. Group normalization is not implemented yet.')

    parser.add_argument('--randomly_drop', default=True, action=argparse.BooleanOptionalAction,
                        help='If we randomly drop modalities during training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for the dataloaders')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')

    parser.add_argument('--validation_interval', type=int, default=5,
                        help='Number of communication rounds before evaluation.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--loss_argument', type=parse_loss_argument, default='dice',
                        help='Loss function and its arguments in the form "loss_name" or "loss_name:param1=value,param2=value"'
                             'dice'
                             'softdice:label_smoothing=0.1'
                             'diceandsoftbce:bce_weight=0.2:dice_weight=0.8:label_smoothing=0.1'
                             'bceloss'
                             'bcewithlogitsloss'
                             'bcediceloss:bce_weight=0.2,dice_weight=0.8')

    parser.add_argument('--BRATS_two_channel_seg', default=False, action=argparse.BooleanOptionalAction,
                        help='if using different segmentation ground truths if both T2 and FLAIR are missing then ground truth is without edema')

    parser.add_argument('--aggregation_method', type=str, default='fedavg', choices=['fedavg', 'fedbn'],
                        help='Specify the aggregation method on the server to use. Current choices: fedavg, fedbn')
    parser.add_argument('--equal_weighting', default=False, action=argparse.BooleanOptionalAction,
                        help='When averaging the parameters, weight each client equally')
    parser.add_argument('--estimate_bn_stats', default=False, action=argparse.BooleanOptionalAction,
                        help='Estimate the running average of mean and variance on clients not present during training')

    parser.add_argument('--training_algorithm', type=str, default='default',
                        choices=['default'],
                        help='Specify the local training algorithm on the clients. Current choices: default')

    parser.add_argument('--wandb_projectname', type=str, default="MT-MRI",
                        help='Project name for wandb logging. If None, no wandb logging is used.')

    parser.add_argument('--resume_training', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to resume training from a checkpoint')

    parser.add_argument('--save_checkpoint', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to save the checkpoint')

    args = parser.parse_args()

    # Create the necessary folders
    output_path = Path('./output') / args.experiment_name
    output_model_path = Path(output_path) / 'models'
    checkpoint_path = output_path / 'checkpoint'
    # Create all necessary folders, if they don't exist
    for f in [output_path, output_model_path, checkpoint_path]:
        os.makedirs(f, exist_ok=True)

    return args


def parse_loss_argument(loss_argument):
    # Split the input string into loss name and arguments
    parts = loss_argument.split(':')
    if len(parts) == 1:
        # No arguments provided
        loss_name, loss_args_str = parts[0], ""
    elif len(parts) == 2:
        # Arguments provided
        loss_name, loss_args_str = parts
    else:
        raise argparse.ArgumentTypeError("Invalid argument format. Use 'loss_name' or 'loss_name:param1=value,param2=value'.")

    # Convert the arguments string to a dictionary
    loss_args = {}
    if loss_args_str:
        try:
            loss_args = {key: float(value) for key, value in (item.split("=") for item in loss_args_str.split(","))}
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid argument format. Use 'param1=value'.")

    return loss_name, loss_args

args = parse_args()
