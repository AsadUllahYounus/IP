import os
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import argparse
from tqdm import tqdm
import logging
import sys
from PIL import Image, ImageOps

# Define exceptions first to avoid circular imports
class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""

class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""

# Utility functions
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Modified GaussianBlur for grayscale OCT images
class GaussianBlur(object):
    """Blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1),
                               stride=1, padding=0, bias=False, groups=1)
        self.blur_v = nn.Conv2d(1, 1, kernel_size=(1, kernel_size),
                               stride=1, padding=0, bias=False, groups=1)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        # Convert the PIL image to a tensor
        img_tensor = self.pil_to_tensor(img).unsqueeze(0)
        
        # For RGB images, convert to grayscale first
        if img_tensor.shape[1] == 3:
            r, g, b = img_tensor[:, 0, :, :], img_tensor[:, 1, :, :], img_tensor[:, 2, :, :]
            img_tensor = 0.299 * r + 0.587 * g + 0.114 * b
            img_tensor = img_tensor.unsqueeze(1)  # Add channel dimension back

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(1, 1)

        self.blur_h.weight.data.copy_(x.view(1, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(1, 1, 1, self.k))

        with torch.no_grad():
            img_tensor = self.blur(img_tensor)
            img_tensor = img_tensor.squeeze()

        # Convert back to PIL
        img = self.tensor_to_pil(img_tensor)
        return img

# OCT-specific adaptive equalization
class AdaptiveEqualization(object):
    """Apply adaptive histogram equalization to enhance contrast in OCT images"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        # Convert to grayscale if not already
        if img.mode != 'L':
            img = ImageOps.grayscale(img)
            
        # Apply adaptive equalization using PIL's built-in methods
        # Since PIL doesn't have CLAHE, we use a simpler equalization
        img = ImageOps.equalize(img)
        return img

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

# Custom OCT dataset class
class OCTImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the OCT images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all image file paths
        self.image_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dcm')):
                    self.image_paths.append(os.path.join(root, file))
        
        # Sort for reproducibility
        self.image_paths.sort()
        
        # Dummy labels (not used for contrastive learning)
        self.labels = [0] * len(self.image_paths)
        
        print(f"Found {len(self.image_paths)} OCT images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Handle DICOM files if present
            if img_path.lower().endswith('.dcm'):
                try:
                    import pydicom
                    ds = pydicom.dcmread(img_path)
                    image = ds.pixel_array
                    # Normalize to 0-255
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                except ImportError:
                    print("Warning: pydicom not installed. DICOM files will be skipped.")
                    # Return a blank grayscale image as a fallback
                    image = Image.new('L', (512, 512), 128)
            else:
                # Open regular image files
                image = Image.open(img_path)
                
                # Convert to grayscale if not already
                if image.mode != 'L':
                    image = ImageOps.grayscale(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank grayscale image as a fallback
            image = Image.new('L', (512, 512), 128)
        
        label = self.labels[idx]  # Dummy label
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_oct_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations optimized for OCT images."""
        # Lighter color jitter for OCT images
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5)
        
        data_transforms = transforms.Compose([
            # Initial equalization to enhance contrast
            AdaptiveEqualization(),
            # Standard SimCLR transformations with OCT-specific parameters
            transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),  # Less aggressive cropping
            transforms.RandomRotation(degrees=10),  # Small rotations to preserve structures
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            GaussianBlur(kernel_size=int(0.1 * size)),
            # Convert to tensor and normalize for grayscale
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalized for grayscale
        ])
        return data_transforms

    def get_dataset(self, name, n_views, image_size=224):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32),
                    n_views),
                download=True
            ),
            'stl10': lambda: datasets.STL10(
                self.root_folder, split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96),
                    n_views),
                download=True
            ),
            'oct': lambda: OCTImageDataset(
                self.root_folder,
                transform=ContrastiveLearningViewGenerator(
                    self.get_oct_pipeline_transform(image_size),
                    n_views)
            )
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection(f"Dataset {name} not supported. Choose from: {list(valid_datasets.keys())}")
        else:
            return dataset_fn()
            
    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Standard SimCLR transform for RGB images (kept for compatibility)"""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

# Model architecture modified for OCT images (grayscale input)
class ResNetOCTSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetOCTSimCLR, self).__init__()
        # Get the ResNet model without pretrained weights
        self.resnet_dict = {
            "resnet18": models.resnet18(weights=None),
            "resnet50": models.resnet50(weights=None)
        }

        self.backbone = self._get_basemodel(base_model)
        
        # Modify the first layer to accept grayscale images (1 channel instead of 3)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get the dimension of the feature vector
        dim_mlp = self.backbone.fc.in_features

        # Add MLP projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), 
            nn.ReLU(), 
            nn.Linear(dim_mlp, out_dim)
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                f"Invalid backbone architecture. Check the config file and pass one of: {list(self.resnet_dict.keys())}")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)  

# Main training class
class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        batch_size = self.args.batch_size
        n_views = self.args.n_views
        
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                # Handle case where the last batch might be smaller than batch_size
                if isinstance(images, list) and len(images) > 0:
                    actual_batch_size = images[0].size(0)
                    if actual_batch_size < self.args.batch_size:
                        continue  # Skip last batch if it's smaller than batch_size
                    
                    images = torch.cat(images, dim=0)
                else:
                    continue  # Skip problematic batches
                
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    # Fix for newer PyTorch versions
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

# Argument parsing and main function
def parse_args():
    model_names = ['resnet18', 'resnet50']
    
    parser = argparse.ArgumentParser(description='PyTorch SimCLR for OCT Images')
    parser.add_argument('-data', metavar='DIR', default='./datasets',
                        help='path to dataset')
    parser.add_argument('-dataset-name', default='oct',
                        help='dataset name', choices=['stl10', 'cifar10', 'oct'])
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64, adjust based on GPU memory)')
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--log-every-n-steps', default=50, type=int,
                        help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--image-size', default=224, type=int, help='Image size for OCT dataset.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Manually set device
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Prepare datasets
    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.image_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    # Create model, optimizer, and scheduler
    model = ResNetOCTSimCLR(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    # Train the model
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)

if __name__ == "__main__":
    main()
