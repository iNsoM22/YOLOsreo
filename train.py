"""
Script for training Regressor Model
"""

from comet_ml import Experiment
from torch.utils import data
from torchvision.models import resnet18
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from script.Model import ResNet18, OrientationLoss
from script.Dataset import Dataset

# Initialize TensorBoard
writer = SummaryWriter()

# Set the device for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Model factories to choose model
model_factory = {
    'resnet18': resnet18(pretrained=True),
}
regressor_factory = {
    'resnet18': ResNet18,
}


def train(
    epochs=10,
    batch_size=32,
    alpha=0.6,
    w=0.4,
    num_workers=2,
    lr=0.0001,
    save_epoch=10,
    train_path=ROOT / 'dataset/KITTI/training',
    model_path=ROOT / 'weights/',
    select_model='resnet18',
    api_key=''
):
    # Directories
    train_path = str(train_path)
    model_path = str(model_path)

    # Load dataset
    print('[INFO] Loading dataset...')
    dataset = Dataset(train_path)

    # Hyperparameters
    hyper_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'w': w,
        'num_workers': num_workers,
        'lr': lr,
        'shuffle': True
    }

    # Comet ML experiment
    experiment = Experiment(api_key, project_name="YOLO3D")
    experiment.log_parameters(hyper_params)

    # Data generator
    data_gen = data.DataLoader(
        dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=hyper_params['shuffle'],
        num_workers=hyper_params['num_workers']
    )

    # Model
    base_model = model_factory[select_model]
    model = regressor_factory[select_model](model=base_model).to(device)

    # Optimizer
    opt_SGD = torch.optim.SGD(
        model.parameters(), lr=hyper_params['lr'], momentum=0.9
    )

    # Loss functions
    conf_loss_func = nn.CrossEntropyLoss().to(device)
    dim_loss_func = nn.MSELoss().to(device)
    orient_loss_func = OrientationLoss

    # Load previous weights
    latest_model = None
    first_epoch = 1
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(
                os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except Exception as e:
            print(f'[ERROR] Could not load latest model: {e}')

    if latest_model is not None:
        checkpoint = torch.load(os.path.join(model_path, latest_model))
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        print(
            f'[INFO] Using previous model {latest_model} at epoch {first_epoch}')
        print('[INFO] Resuming training...')

    total_num_batches = len(data_gen)

    with experiment.train():
        for epoch in range(first_epoch, int(hyper_params['epochs']) + 1):
            curr_batch = 0
            passes = 0
            with tqdm(data_gen, unit='batch') as tepoch:
                for local_batch, local_labels in tepoch:
                    # Progress bar
                    tepoch.set_description(f'Epoch {epoch}')

                    # Ground-truth
                    truth_orient = local_labels['Orientation'].float().to(
                        device)
                    truth_conf = local_labels['Confidence'].float().to(device)
                    truth_dim = local_labels['Dimensions'].float().to(device)

                    # Convert to CUDA
                    local_batch = local_batch.float().to(device)

                    # Forward pass
                    orient, conf, dim = model(local_batch)

                    # Loss calculations
                    orient_loss = orient_loss_func(
                        orient, truth_orient, truth_conf)
                    dim_loss = dim_loss_func(dim, truth_dim)

                    # Ensure the indices are correct
                    truth_conf = torch.max(truth_conf, dim=1)[1]
                    conf_loss = conf_loss_func(conf, truth_conf)

                    loss_theta = conf_loss + w * orient_loss
                    loss = alpha * dim_loss + loss_theta

                    # Log metrics
                    writer.add_scalar('Loss/train', loss.item(), epoch)
                    experiment.log_metric(
                        'Loss/train', loss.item(), epoch=epoch)

                    opt_SGD.zero_grad()
                    loss.backward()
                    opt_SGD.step()

                    # Progress bar update
                    tepoch.set_postfix(loss=loss.item())

            # Save model periodically
            if epoch % save_epoch == 0:
                model_name = os.path.join(
                    model_path, f'{select_model}_epoch_{epoch}.pkl')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss.item()
                }, model_name)
                print(f'[INFO] Saving weights as {model_name}')

    writer.flush()
    writer.close()


def parse_opt():
    parser = argparse.ArgumentParser(description='Regressor Model Training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha (default=0.6, do not change)')
    parser.add_argument('--w', type=float, default=0.4,
                        help='Weight (do not change)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers (for colab & kaggle, use 2)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='Save model every # epochs')
    parser.add_argument('--train_path', type=str, default=ROOT /
                        'dataset/KITTI/training', help='Training path for KITTI dataset')
    parser.add_argument('--model_path', type=str, default=ROOT /
                        'weights', help='Weights path for loading and saving model')
    parser.add_argument('--select_model', type=str, default='resnet18',
                        help='Model selection: {resnet18}')
    parser.add_argument('--api_key', type=str, default='',
                        help='API key for comet.ml')

    opt = parser.parse_args()
    return opt


def main(opt):
    train(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
