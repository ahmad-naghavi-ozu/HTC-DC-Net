import os
import torch
import torch.utils.data
import numpy as np
from skimage import io
import random
import glob
from tqdm import tqdm
import torchvision.transforms as tfs

class DFC2023SDataset(torch.utils.data.Dataset):
    """
    Dataset class for DFC2023S dataset.
    
    Structure:
    DFC2023S/
    ├── test/
    │   ├── dsm/  (Digital Surface Model data - elevation information)
    │   ├── rgb/  (RGB optical imagery)
    │   ├── sar/  (Synthetic Aperture Radar imagery)
    │   └── sem/  (Semantic segmentation masks/labels)
    ├── train/
    │   ├── dsm/
    │   ├── rgb/
    │   ├── sar/
    │   └── sem/
    └── valid/
        ├── dsm/
        ├── rgb/
        ├── sar/
        └── sem/
    """
    def __init__(self, data_dir, split='train', image_size=256, use_mask=True,
                crop=None, hflip=False, normalize=True, use_sar=False):
        super(DFC2023SDataset, self).__init__()
        self.root = data_dir
        self.split = split
        self.image_size = image_size
        self.crop = crop
        self.hflip = hflip
        self.normalize = normalize
        self.use_mask = use_mask
        self.use_sar = use_sar
        
        # Get paths of all images in the given split
        self.paths = self._collect_files()
        self.size = len(self.paths)
        
        # Load or compute image stats for normalization
        self.stats_file = os.path.join(data_dir, f'dfc2023s_stats.pickle')
        if os.path.exists(self.stats_file):
            self.image_mean, self.image_std, self.dsm_mean, self.dsm_std, _, self.dsm_max = torch.load(self.stats_file)
        else:
            self.compute_dataset_stats()
    
    def _collect_files(self):
        """Collect all files for the specified split."""
        split_dir = os.path.join(self.root, self.split)
        rgb_files = sorted(glob.glob(os.path.join(split_dir, 'rgb', '*.tif')))
        
        files = []
        for rgb_path in rgb_files:
            filename = os.path.basename(rgb_path)
            base_name = os.path.splitext(filename)[0]
            
            dsm_path = os.path.join(self.root, self.split, 'dsm', f"{base_name}.tif")
            if not os.path.exists(dsm_path):
                continue  # Skip if no DSM data
                
            sem_path = os.path.join(self.root, self.split, 'sem', f"{base_name}.tif")
            if self.use_mask and not os.path.exists(sem_path):
                continue  # Skip if mask is required but not available
                
            sar_path = os.path.join(self.root, self.split, 'sar', f"{base_name}.tif") if self.use_sar else None
            
            files.append((rgb_path, dsm_path, sem_path, sar_path))
            
        return files
    
    def compute_dataset_stats(self):
        """Compute dataset statistics for normalization."""
        print(f"Computing statistics for DFC2023S dataset...")
        
        # Image stats
        sum_x = np.zeros(3)
        sum_x2 = np.zeros(3)
        sum_size = 0
        
        # DSM stats
        sum_dsm = 0
        sum_dsm2 = 0
        sum_dsm_size = 0
        dsm_min = float('inf')
        dsm_max = float('-inf')
        
        for rgb_path, dsm_path, _, _ in tqdm(self.paths):
            # Image stats
            img = io.imread(rgb_path).astype(np.float32)
            if len(img.shape) == 2:  # Handle grayscale images
                img = np.stack([img, img, img], axis=2)
            
            img = img.reshape(-1, 3)
            sum_x += img.sum(axis=0)
            sum_x2 += (img * img).sum(axis=0)
            sum_size += img.shape[0]
            
            # DSM stats
            dsm = io.imread(dsm_path).astype(np.float32)
            dsm = np.nan_to_num(dsm)
            dsm_flat = dsm.flatten()
            
            sum_dsm += dsm_flat.sum()
            sum_dsm2 += (dsm_flat ** 2).sum()
            sum_dsm_size += dsm_flat.size
            
            dsm_min = min(dsm_min, dsm_flat.min())
            dsm_max = max(dsm_max, dsm_flat.max())
        
        # Calculate means and stds
        image_mean = sum_x / sum_size
        image_std = np.sqrt(sum_x2 / sum_size - image_mean * image_mean)
        
        dsm_mean = sum_dsm / sum_dsm_size
        dsm_std = np.sqrt(sum_dsm2 / sum_dsm_size - dsm_mean * dsm_mean)
        
        # Set DSM min to 0 if negative
        dsm_min = max(0, dsm_min)
        
        # Save statistics
        torch.save([image_mean, image_std, dsm_mean, dsm_std, dsm_min, dsm_max], self.stats_file)
        
        self.image_mean = image_mean
        self.image_std = image_std
        self.dsm_mean = dsm_mean
        self.dsm_std = dsm_std
        self.dsm_min = dsm_min
        self.dsm_max = dsm_max
        
        print(f"Image mean: {image_mean}, std: {image_std}")
        print(f"DSM mean: {dsm_mean}, std: {dsm_std}, min: {dsm_min}, max: {dsm_max}")
    
    def transform_image(self, img):
        """Apply transformations to the input image."""
        img = torch.tensor(img.astype(np.float32).transpose(2, 0, 1))
        
        if self.crop:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        
        if self.hflip and random.random() > 0.5:
            img = tfs.functional.hflip(img)
        
        if self.normalize:
            img = tfs.functional.normalize(img, mean=self.image_mean, std=self.image_std)
        
        return img
    
    def __getitem__(self, index):
        rgb_path, dsm_path, sem_path, sar_path = self.paths[index]
        
        # Use the file basename as index
        file_idx = os.path.basename(rgb_path).split('.')[0]
        
        # Load RGB image
        image = io.imread(rgb_path)
        if len(image.shape) == 2:  # Handle grayscale images
            image = np.stack([image, image, image], axis=2)
        
        # Load DSM (height data)
        dsm = np.nan_to_num(io.imread(dsm_path).astype(np.float32)).clip(0)
        
        # Prepare ground truth dictionary
        gt_dict = {"ndsm": torch.tensor(dsm)[None, :, :]}
        
        # Load semantic mask if available
        if self.use_mask and sem_path:
            mask = io.imread(sem_path)
            # Create building mask: buildings are typically class 1 in semantic segmentation
            building_mask = np.float32(mask == 1)  # Adjust class ID if needed
            gt_dict.update({"mask": torch.tensor(building_mask)[None, :, :]})
        
        # Load SAR data if requested
        if self.use_sar and sar_path:
            sar_data = io.imread(sar_path)
            if len(sar_data.shape) == 2:
                sar_data = np.expand_dims(sar_data, axis=2)
            image = np.concatenate([image, sar_data], axis=2)
        
        # Apply transformations
        image_tensor = self.transform_image(image)
        
        return file_idx, image_tensor, gt_dict
    
    def __len__(self):
        return self.size
    
    def name(self):
        return 'DFC2023SDataset'


def create_dfc2023s_dataloaders(cfgs):
    """Create dataloaders for training and validation."""
    batch_size = cfgs.get('batch_size', 8)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)
    use_mask = cfgs.get('use_mask', True)
    normalize = cfgs.get('normalize', True)
    hflip = cfgs.get('hflip', False)
    use_sar = cfgs.get('use_sar', False)
    
    # Check if using mini version for debugging
    is_mini = cfgs.get('use_mini_dataset', False)
    if is_mini:
        data_dir = cfgs.get('mini_dfc2023s_dir', '/home/asfand/Ahmad/datasets/DFC2023Amini/')
        print(f"Using minimized dataset for debugging: {data_dir}")
    else:
        data_dir = cfgs.get('dfc2023s_dir', '/home/asfand/Ahmad/datasets/DFC2023S/')
    
    train_dataset = DFC2023SDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        use_mask=use_mask,
        crop=crop,
        hflip=hflip,
        normalize=normalize,
        use_sar=use_sar
    )
    
    val_dataset = DFC2023SDataset(
        data_dir=data_dir,
        split='valid',
        image_size=image_size,
        use_mask=use_mask,
        crop=crop,
        hflip=False,  # No flipping for validation
        normalize=normalize,
        use_sar=use_sar
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_dfc2023s_test_dataloaders(cfgs):
    """Create dataloaders for testing."""
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)
    normalize = cfgs.get('normalize', True)
    use_mask = cfgs.get('test_use_mask', True)
    use_sar = cfgs.get('use_sar', False)
    
    # Check if using mini version for debugging
    is_mini = cfgs.get('use_mini_dataset', False)
    if is_mini:
        data_dir = cfgs.get('mini_dfc2023s_dir', '/home/asfand/Ahmad/datasets/DFC2023Amini/')
        print(f"Using minimized dataset for testing: {data_dir}")
    else:
        data_dir = cfgs.get('dfc2023s_dir', '/home/asfand/Ahmad/datasets/DFC2023S/')
    
    test_dataset = DFC2023SDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        use_mask=use_mask,
        crop=crop,
        hflip=False,
        normalize=normalize,
        use_sar=use_sar
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {"DFC2023S_test": test_loader}