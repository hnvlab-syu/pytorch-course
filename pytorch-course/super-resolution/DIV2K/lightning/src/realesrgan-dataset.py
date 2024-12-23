import os
import glob
import numpy as np
import torch
import lightning as L

import random
from torch.nn import functional as F

from PIL import Image
from typing import Optional
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from basicsr.data.transforms import augment, paired_random_crop

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D


SEED = 36
L.seed_everything(SEED)


class DIV2KDataModule(L.LightningDataModule):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        mode: str,
        upscale: int,
    ):
        super().__init__()
        self.mode = mode
        if self.mode == 'train':    # train(val, test) or prediction
            self.lr_dataset = sorted(glob.glob(os.path.join(lr_dir, "*.png")))   # */*.png   # list format
            self.hr_dataset = sorted(glob.glob(os.path.join(hr_dir, "*.png")))

            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            if not self.lr_dataset or not self.hr_dataset:
                raise RuntimeError(f"No images found in {lr_dir} or {hr_dir}")
            if len(self.lr_dataset) != len(self.hr_dataset):
                raise RuntimeError("Number of LR and HR images don't match")
        else:
            self.data_path = lr_dir     # example.jpg 경로
            batch_size = 1

            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.upscale = upscale

        self.gt_size = 160
        self.use_hflip = True
        self.use_rot = True 

    def setup(self, stage: Optional[str] = None):
        if self.mode == 'train':
            # x: lr, y: hr
            train_x, val_x, train_y, val_y = train_test_split(self.lr_dataset, self.hr_dataset, test_size=0.2, random_state=SEED)
            val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.5, random_state=SEED)

            train_data = [(x, y) for x, y in zip(train_x, train_y)]
            val_data = [(x, y) for x, y in zip(val_x, val_y)]
            test_data = [(x, y) for x, y in zip(test_x, test_y)]
        else:
            pred_data = [Image.open(self.data_path).convert('RGB')]

        # trainer
        if stage == "fit":
            self.train_dataset = train_data
            self.val_dataset = val_data
            print('---------------------')
            print(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

        if stage == "test":
            self.test_dataset = test_data
            print('---------------------')
            print(f"Test size: {len(self.test_dataset)}")

        if stage == "predict":
            self.pred_dataset = pred_data


    # def _train_collate_fn(self, batch):
    #     lr_list = []
    #     hr_list = []
    #     scale = self.upscale

    #     for lr_img, hr_img in batch:
    #         lr_img= Image.open(lr_img).convert('RGB')
    #         hr_img = Image.open(hr_img).convert('RGB')

    #         # PIL Image-> numpy array: BasicSR의 paired_random_crop(), augment()의 입력 형식을 맞추기 위함
    #         lr_np = np.array(lr_img) / 255.0  # [0, 1]로 정규화
    #         hr_np = np.array(hr_img) / 255.0

    #         # Augmentation: random crop
    #         gt_size = self.gt_size
    #         hr_np, lr_np = paired_random_crop(hr_np, lr_np, gt_size, scale)

    #         # Augmentation: flip, rotation
    #         hr_np, lr_np = augment([hr_np, lr_np], hflip=True, rotation=True)

    #         # numpy array-> PIL Image
    #         lr_img = Image.fromarray((lr_np * 255).astype(np.uint8))
    #         hr_img = Image.fromarray((hr_np * 255).astype(np.uint8))

    #         # transform (totensor)
    #         if self.transform:
    #             lr_img = self.transform(lr_img)
    #             hr_img = self.transform(hr_img)

    #         lr_list.append(lr_img)
    #         hr_list.append(hr_img)

    #     return torch.stack(lr_list), torch.stack(hr_list)
    

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)


    def _predict_collate_fn(self, batch):
        img = batch[0]
        input = self.transform(img).unsqueeze(0)
        return input
    

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,  
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            collate_fn=self.feed_data,
            shuffle=True)
                        

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            collate_fn=self.feed_data)


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=self.feed_data)


    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset, 
            batch_size=self.batch_size,   # 1 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=self._predict_collate_fn)

