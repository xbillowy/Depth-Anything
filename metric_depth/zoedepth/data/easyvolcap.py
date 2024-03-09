import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from glob import glob
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

from zoedepth.utils.data_utils import rotate_image

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.math_utils import affine_inverse, affine_padding, torch_inverse_3x3, point_padding
from easyvolcap.utils.bound_utils import get_bound_2d_bound, get_bounds, monotonic_near_far, get_bound_3d_near_far
from easyvolcap.utils.data_utils import DataSplit, UnstructuredTensors, load_resize_undist_ims_bytes, load_image_from_bytes, as_torch_func, to_cuda, to_cpu, to_tensor, export_pts, load_pts, decode_crop_fill_ims_bytes, decode_fill_ims_bytes


class EasyVolcap(Dataset):
    def __init__(self,
                 config
                 ):

        self.dataset = config.dataset
        self.split = config.split
        # Get the paths to the data, modify them in the config file `metric_depth/zoedepth/utils/config.py`
        self.data_root = config.data_root
        self.intri_file = config.intri_file
        self.extri_file = config.extri_file
        self.images_dir = config.images_dir
        self.depths_dir = config.depths_dir
        self.cameras_dir = config.cameras_dir
        self.masks_dir = config.masks_dir

        # The frame number & image size should be inferred from the dataset
        self.view_sample = config.view_sample
        self.frame_sample = config.frame_sample
        if self.view_sample[1] is not None: self.n_view_total = self.view_sample[1]
        else: self.n_view_total = len(os.listdir(join(self.data_root, self.images_dir)))  # total number of cameras before filtering
        if self.frame_sample[1] is not None: self.n_frames_total = self.frame_sample[1]
        else: self.n_frames_total = min([len(glob(join(self.data_root, self.images_dir, cam, '*'))) for cam in os.listdir(join(self.data_root, self.images_dir))])  # total number of images before filtering

        # Compute needed visual hulls & align all cameras loaded
        self.load_cameras()  # load and normalize all cameras (center lookat, align y axis)
        self.select_cameras()  # select repective cameras to use

        # Images and masks and depths related stuff
        self.use_masks = config.use_masks
        self.use_depths = config.use_depths
        self.ims_pattern = config.ims_pattern

        self.dist_mask = config.dist_mask
        self.ratio = config.ratio
        self.imsize_overwrite = config.imsize_overwrite
        self.center_crop_size = config.center_crop_size
        self.dist_opt_K = config.dist_opt_K
        self.encode_ext = config.encode_ext
        self.cache_raw = config.cache_raw

        # Load the image paths and the corresponding depth paths
        self.load_paths()  # load image files into self.ims
        # Load the images and the corresponding depths
        self.load_bytes()

        # Data augmentationn related stuff
        self.augment = config.get('aug', True)  # default True
        self.random_crop = config.get('random_crop', False)  # default False
        self.crop_size = config.get('crop_size', None)  # default None
        self.random_translate = config.get('random_translate', False)  # default False
        self.translate_maxl = config.get('translate_maxl', None)  # default None
        self.translate_prob = config.get('translate_prob', 0.20)  # default 0.20
        self.random_rotate = config.get('do_random_rotate', True)  # default True
        self.rotate_degree = config.get('rotate_degree', 1)  # default 1

        # Data preprocessing related stuff
        self.do_normalize = config.get('do_normalize', False)  # default False
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_cameras(self):
        # Load camera related stuff like image list and intri, extri.
        # Determine whether it is a monocular dataset or multiview dataset based on the existence of root `extri.yml` or `intri.yml`
        # Multiview dataset loading, need to expand, will have redundant information
        if exists(join(self.data_root, self.intri_file)) and exists(join(self.data_root, self.extri_file)):
            self.cameras = read_camera(join(self.data_root, self.intri_file), join(self.data_root, self.extri_file))
            self.camera_names = np.asarray(sorted(list(self.cameras.keys())))  # NOTE: sorting camera names
            self.cameras = dotdict({k: [self.cameras[k] for i in range(self.n_frames_total)] for k in self.camera_names})
            # TODO: Handle avg processing

        # Monocular dataset loading, each camera has a separate folder
        elif exists(join(self.data_root, self.cameras_dir)):
            self.camera_names = np.asarray(sorted(os.listdir(join(self.data_root, self.cameras_dir))))  # NOTE: sorting here is very important!
            self.cameras = dotdict({
                k: [v[1] for v in sorted(
                    read_camera(join(self.data_root, self.cameras_dir, k, self.intri_file),
                                join(self.data_root, self.cameras_dir, k, self.extri_file)).items()
                )] for k in self.camera_names
            })
            # TODO: Handle avg export and loading for such monocular dataset
        else:
            raise NotImplementedError(f'Could not find {{{self.intri_file},{self.extri_file}}} or {self.cameras_dir} directory in {self.data_root}, check your dataset configuration')

        # Expectation:
        # self.camera_names: a list containing all camera names
        # self.cameras: a mapping from camera names to a list of camera objects
        # (every element in list is an actual camera for that particular view and frame)
        # NOTE: ALWAYS, ALWAYS, SORT CAMERA NAMES.
        self.Hs = torch.as_tensor([[cam.H for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ws = torch.as_tensor([[cam.W for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ks = torch.as_tensor([[cam.K for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Rs = torch.as_tensor([[cam.R for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Ts = torch.as_tensor([[cam.T for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 1
        self.Ds = torch.as_tensor([[cam.D for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 1, 5
        self.ts = torch.as_tensor([[cam.t for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.ns = torch.as_tensor([[cam.n for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.fs = torch.as_tensor([[cam.f for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Cs = -self.Rs.mT @ self.Ts  # V, F, 3, 1
        self.w2cs = torch.cat([self.Rs, self.Ts], dim=-1)  # V, F, 3, 4
        self.c2ws = affine_inverse(self.w2cs)  # V, F, 3, 4

    def select_cameras(self):
        # Only retrain needed
        # Perform view selection first
        view_inds = torch.arange(self.Ks.shape[0])
        if len(self.view_sample) != 3: view_inds = view_inds[self.view_sample]  # this is a list of indices
        else: view_inds = view_inds[self.view_sample[0]:self.view_sample[1]:self.view_sample[2]]  # begin, start, end
        self.view_inds = view_inds
        if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # Perform frame selection next
        frame_inds = torch.arange(self.Ks.shape[1])
        if len(self.frame_sample) != 3: frame_inds = frame_inds[self.frame_sample]
        else: frame_inds = frame_inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.frame_inds = frame_inds  # used by `load_smpls()`
        if len(frame_inds) == 1: frame_inds = [frame_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # NOTE: if view_inds == [0,] in monocular dataset or whatever case, type(`self.camera_names[view_inds]`) == str, not a list of str
        self.camera_names = np.asarray([self.camera_names[view] for view in view_inds])  # this is what the b, e, s means
        self.cameras = dotdict({k: [self.cameras[k][int(i)] for i in frame_inds] for k in self.camera_names})  # reloading
        self.Hs = self.Hs[view_inds][:, frame_inds]
        self.Ws = self.Ws[view_inds][:, frame_inds]
        self.Ks = self.Ks[view_inds][:, frame_inds]
        self.Rs = self.Rs[view_inds][:, frame_inds]
        self.Ts = self.Ts[view_inds][:, frame_inds]
        self.Ds = self.Ds[view_inds][:, frame_inds]
        self.ts = self.ts[view_inds][:, frame_inds]
        self.Cs = self.Cs[view_inds][:, frame_inds]
        self.w2cs = self.w2cs[view_inds][:, frame_inds]
        self.c2ws = self.c2ws[view_inds][:, frame_inds]

    def load_paths(self):
        # Load image related stuff for reading from disk later
        # If number of images in folder does not match, here we'll get an error
        ims = [[join(self.data_root, self.images_dir, cam, self.ims_pattern.format(frame=i)) for i in range(self.n_frames_total)] for cam in self.camera_names]
        if not exists(ims[0][0]):
            ims = [[i.replace('.' + self.ims_pattern.split('.')[-1], '.JPG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.JPG', '.png') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.png', '.PNG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [sorted(glob(join(self.data_root, self.images_dir, cam, '*')))[:self.n_frames_total] for cam in self.camera_names]
        ims = [np.asarray(ims[i])[:min([len(i) for i in ims])] for i in range(len(ims))]  # deal with the fact that some weird dataset has different number of images
        self.ims = np.asarray(ims)  # V, N
        self.ims_dir = join(*split(dirname(self.ims[0, 0]))[:-1])  # logging only

        # TypeError: can't convert np.ndarray of type numpy.str_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        # MARK: Names stored as np.ndarray
        inds = np.arange(self.ims.shape[-1])
        if len(self.frame_sample) != 3: inds = inds[self.frame_sample]
        else: inds = inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.ims = self.ims[..., inds]  # these paths are later used for reading images from disk

        # Mask path preparation
        if self.use_masks:
            self.mks = np.asarray([im.replace(self.images_dir, self.masks_dir) for im in self.ims.ravel()]).reshape(self.ims.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('.png', '.jpg') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('.jpg', '.png') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):  # Two types of commonly used mask directories
                self.mks = np.asarray([mk.replace(self.masks_dir, 'masks') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('masks', 'mask') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('masks', 'msk') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            self.mks_dir = join(*split(dirname(self.mks[0, 0]))[:-1])

        # Depth image path preparation
        if self.use_depths:
            self.dps = np.asarray([im.replace(self.images_dir, self.depths_dir).replace('.jpg', '.exr').replace('.png', '.exr') for im in self.ims.ravel()]).reshape(self.ims.shape)
            if not exists(self.dps[0, 0]):
                self.dps = np.asarray([dp.replace('.exr', 'exr') for dp in self.dps.ravel()]).reshape(self.dps.shape)
            self.dps_dir = join(*split(dirname(self.dps[0, 0]))[:-1])  # logging only

    def load_bytes(self):
        # Camera distortions are only applied on the ground truth image, the rendering model does not include these
        # And unlike intrinsic parameters, it has no direct dependency on the size of the loaded image, thus we directly process them here
        dist_mask = torch.as_tensor(self.dist_mask)
        self.Ds = self.Ds.view(*self.Ds.shape[:2], 5) * dist_mask  # some of the distortion parameters might need some manual massaging

        # Need to convert to a tight data structure for access
        ori_Ks = self.Ks
        ori_Ds = self.Ds
        ratio = self.imsize_overwrite if self.imsize_overwrite[0] > 0 else self.ratio  # maybe force size, or maybe use ratio to resize
        if self.use_masks:
            self.mks_bytes, self.Ks, self.Hs, self.Ws = \
                load_resize_undist_ims_bytes(self.mks, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                             f'Loading mask bytes for {blue(self.mks_dir)} {magenta(self.split)}',
                                             decode_flag=cv2.IMREAD_GRAYSCALE, dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)  # will for a grayscale read from bytes
            self.Ks = torch.as_tensor(self.Ks)
            self.Hs = torch.as_tensor(self.Hs)
            self.Ws = torch.as_tensor(self.Ws)

        # Maybe load depth images here, using HDR
        if self.use_depths:  # TODO: implement HDR loading
            self.dps_bytes, self.Ks, self.Hs, self.Ws = \
                load_resize_undist_ims_bytes(self.dps, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                             f'Loading dpts bytes for {blue(self.dps_dir)} {magenta(self.split)}',
                                             decode_flag=cv2.IMREAD_UNCHANGED, dist_opt_K=self.dist_opt_K, encode_ext='.exr')  # will for a grayscale read from bytes

        # Image pre cacheing (from disk to memory)
        self.ims_bytes, self.Ks, self.Hs, self.Ws = \
            load_resize_undist_ims_bytes(self.ims, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                         f'Loading imgs bytes for {blue(self.ims_dir)} {magenta(self.split)}',
                                         dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        self.Ks = torch.as_tensor(self.Ks)
        self.Hs = torch.as_tensor(self.Hs)
        self.Ws = torch.as_tensor(self.Ws)

        if self.cache_raw:
            # To make memory access faster, store raw floats in memory
            self.ims_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.ims_bytes, desc=f'Caching imgs for {blue(self.data_root)} {magenta(self.split)}')])  # High mem usage
            if hasattr(self, 'mks_bytes'): self.mks_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.mks_bytes, desc=f'Caching mks for {blue(self.data_root)} {magenta(self.split)}')])
            if hasattr(self, 'dps_bytes'): self.dps_bytes = to_tensor([load_image_from_bytes(x, normalize=False) for x in tqdm(self.dps_bytes, desc=f'Caching dps for {blue(self.data_root)} {magenta(self.split)}')])

    @property
    def n_views(self): return len(self.cameras)

    @property
    def n_latents(self): return len(next(iter(self.cameras.values())))  # short for timestamp

    # NOTE: everything beginning with get are utilities for __getitem__
    # NOTE: coding convension are preceded with "NOTE"
    def get_indices(self, index):
        # These indices are relative to the processed dataset
        view_index, latent_index = index // self.n_latents, index % self.n_latents

        if len(self.view_sample) != 3: camera_index = self.view_sample[view_index]
        else: camera_index = view_index * self.view_sample[2] + self.view_sample[0]

        if len(self.frame_sample) != 3: frame_index = self.frame_sample[latent_index]
        else: frame_index = latent_index * self.frame_sample[2] + self.frame_sample[0]

        return view_index, latent_index, camera_index, frame_index

    def get_image_bytes(self, view_index: int, latent_index: int):
        im_bytes = self.ims_bytes[view_index * self.n_latents + latent_index]  # MARK: no fancy indexing
        if self.use_masks:
            mk_bytes = self.mks_bytes[view_index * self.n_latents + latent_index]
            wt_bytes = mk_bytes.clone()
        else:
            mk_bytes, wt_bytes = None, None

        if self.use_depths:
            dp_bytes = self.dps_bytes[view_index * self.n_latents + latent_index]
        else:
            dp_bytes = None

        return im_bytes, mk_bytes, wt_bytes, dp_bytes

    def get_image(self, view_index: int, latent_index: int):
        # Load bytes (rgb, msk, wet, dpt)
        im_bytes, mk_bytes, wt_bytes, dp_bytes = self.get_image_bytes(view_index, latent_index)
        rgb, msk, wet, dpt = None, None, None, None

        # Load image from bytes
        if self.cache_raw:
            rgb = torch.as_tensor(im_bytes)
        else:
            rgb = torch.as_tensor(load_image_from_bytes(im_bytes, normalize=True))  # 4-5ms for 400 * 592 jpeg, sooo slow

        # Load mask from bytes
        if mk_bytes is not None:
            if self.cache_raw:
                msk = torch.as_tensor(mk_bytes)
            else:
                msk = torch.as_tensor(load_image_from_bytes(mk_bytes, normalize=True)[..., :1])
        else:
            msk = torch.ones_like(rgb[..., -1:])

        # Load sampling weights from bytes
        if wt_bytes is not None:
            if self.cache_raw:
                wet = torch.as_tensor(wt_bytes)
            else:
                wet = torch.as_tensor(load_image_from_bytes(wt_bytes, normalize=True)[..., :1])
        else:
            wet = msk.clone()

        # Load depth from bytes
        if dp_bytes is not None:
            if self.cache_raw:
                dpt = torch.as_tensor(dp_bytes)
            else:
                dpt = torch.as_tensor(load_image_from_bytes(dp_bytes, normalize=False)[..., :1])  # readin as is

        return rgb, msk, wet, dpt

    def __getitem__(self, idx):
        # Prepare the output data
        data = dict()
        data['dataset'] = self.dataset

        # Load the indices
        view_index, latent_index, camera_index, frame_index = self.get_indices(idx)
        data['view_index'], data['latent_index'], data['camera_index'], data['frame_index'] = view_index, latent_index, camera_index, frame_index

        # Load the camera parameters
        w2c = self.w2cs[view_index, latent_index]  # 3, 4
        c2w = self.c2ws[view_index, latent_index]  # 3, 4
        ixt = self.Ks[view_index, latent_index]  # 3, 3
        data['w2c'], data['c2w'], data['ixt'] = w2c, c2w, ixt
        data['H'], data['W'] = self.Hs[view_index, latent_index], self.Ws[view_index, latent_index]

        # Load the rgb image, depth and mask
        rgb, msk, _, dpt = self.get_image(view_index, latent_index)
        # Perform augmentation
        rgb, dpt, msk = self.perform_augment(rgb, dpt, msk)  # H, W, 3; H, W, 1; H, W, 1
        # Deal with the order of the channels
        data['image'] = rgb.permute(2, 0, 1)  # 3, H, W
        if dpt is not None: data['depth'] = dpt.permute(2, 0, 1)  # 1, H, W
        if msk is not None: data['mask']  = msk.permute(2, 0, 1)  # 1, H, W
        # Perform normalization if needed
        if self.do_normalize: data['image'] = self.normalize(data['image'])

        # Record the paths?
        data['image_path'] = self.ims[view_index, latent_index]
        if dpt is not None: data['depth_path'] = self.dps[view_index, latent_index]

        return data

    def __len__(self):
        return self.n_views * self.n_latents  # there's no notion of epoch here

    def perform_augment(self, img, dpt, msk):
        # Return directly if no augmentation is needed
        if self.split != 'train' or not self.augment: return img, dpt, msk

        # Convert from torch.Tensor to numpy.ndarray first
        img = img.numpy()  # H, W, 3
        if dpt is not None: dpt = dpt.numpy()  # H, W, 1
        if msk is not None: msk = msk.numpy()  # H, W, 1

        # Perform different augmentations
        if self.random_rotate:
            img, dpt, msk = self.perform_random_rotate(img, dpt, msk, self.rotate_degree)
        if self.random_crop:
            img, dpt, msk = self.perform_random_crop(img, dpt, msk, self.crop_size)
        if self.random_translate:
            img, dpt, msk = self.perform_random_translate(img, dpt, msk, self.translate_maxl, self.translate_prob)

        # Perform random flip
        img, dpt, msk = self.perform_random_flip(img, dpt, msk)
        # Perform gamma, brightness and color augmentation
        img = self.perform_random_augment(img)

        # Convert from numpy.ndarray to torch.Tensor
        img = torch.as_tensor(img, dtype=torch.float32)  # H, W, 3
        if dpt is not None: dpt = torch.as_tensor(dpt, dtype=torch.float32)  # H, W, 1
        if msk is not None: msk = torch.as_tensor(msk, dtype=torch.float32)  # H, W, 1

        return img, dpt, msk

    def perform_random_rotate(self, img, dpt, msk, rotate_degree=1):
        # Get rotate angle
        rotate_angle = (random.random() - 0.5) * 2 * rotate_degree

        # Perform random rotate
        img = rotate_image(img, rotate_angle)
        if dpt is not None: dpt = rotate_image(dpt, rotate_angle)
        if msk is not None: msk = rotate_image(msk, rotate_angle)

        return img, dpt, msk

    def perform_random_flip(self, img, dpt, msk):
        # Return directly if no flip is needed
        if random.random() < 0.5: return img, dpt, msk

        # Perform flip
        img = img[:, ::-1, :].copy()
        if dpt is not None: dpt = dpt[:, ::-1, :].copy()
        if msk is not None: msk = msk[:, ::-1, :].copy()

        return img, dpt, msk

    def perform_random_augment(self, img):
        # Return directly if no augmentation is needed
        if random.random() < 0.5: return img

        # Perform gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        img = img ** gamma

        # Perform brightness augmentation
        brightness = random.uniform(0.8, 1.2)
        img = img * brightness

        # Perform color augmentation
        color = np.random.uniform(0.1, 1.1, size=3)  # (3,)
        white = np.ones(img.shape[:2])  # (H, W)
        color = np.stack([white * color[i] for i in range(len(color))], axis=-1)  # (H, W, 3)
        img = img * color
        img = np.clip(img, 0, 1)

        return img

    def perform_random_crop(self, img, dpt, msk, crop_size=None):
        # Return directly if no random crop is needed
        assert crop_size is not None, 'The crop size should be specified'

        Ho, Wo = img.shape[:2]
        Hc, Wc = crop_size
        # Perform random crop
        x, y = random.randint(0, Wo - Wc), random.randint(0, Ho - Hc)
        img = img[y:y + Hc, x:x + Wc, :]
        if dpt is not None: dpt = dpt[y:y + Hc, x:x + Wc]
        if msk is not None: msk = msk[y:y + Hc, x:x + Wc]

        return img, dpt, msk

    def perform_random_translate(self, img, dpt, msk, translate_maxl=None, translate_prob=0.20):
        # Return directly if no random translate is needed
        assert translate_maxl is not None, 'The maximum translation length should be specified'

        # Return directly if the random translate is not performed
        if random.random() > translate_prob or translate_maxl <= 0: return img, dpt, msk

        # Perform random translation
        H, W = img.shape[:2]
        x, y = random.randint(-translate_maxl, translate_maxl), random.randint(-translate_maxl, translate_maxl)
        M = np.float32([[1, 0, x], [0, 1, y]])
        img = cv2.warpAffine(img, M, (W, H))
        if dpt is not None: dpt = cv2.warpAffine(dpt, M, (W, H))
        if msk is not None: msk = cv2.warpAffine(msk, M, (W, H))

        return img, dpt, msk


def get_easyvolcap_loader(config, batch_size=1, mode='train', **kwargs):
    # FIXME: find a better way to handle EasyVolcap test dataset
    if mode == 'online_eval' and 'test' not in config.dataset and 'visualize' not in config.dataset:
        config.split = 'test'
        config.view_sample = [0, None, 660]
        config.frame_sample = [0, None, 1]

    # Build the dataloader
    dataloader = DataLoader(EasyVolcap(config), batch_size=batch_size, **kwargs)

    return dataloader
