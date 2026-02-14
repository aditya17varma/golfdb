import os.path as osp
import os
import time
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        nested_vid_dir = osp.join(vid_dir, osp.basename(osp.normpath(vid_dir)))
        self.vid_dir_candidates = [vid_dir]
        if osp.isdir(nested_vid_dir):
            self.vid_dir_candidates.append(nested_vid_dir)
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.debug = os.getenv('GOLFDB_DEBUG_DATALOADER', '0') == '1'
        self._resolved_vid_dir = None

    def _log(self, msg):
        if not self.debug:
            return
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 'main'
        print('[GolfDB][worker={}][pid={}] {}'.format(worker_id, os.getpid(), msg), flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        t0 = time.time()
        cap = None
        try:
            a = self.df.loc[idx, :]  # annotation info
            events = np.asarray(a['events']).copy()
            events -= events[0]  # now frame #s correspond to frames in preprocessed video clips
            event_frames = events[1:-1]
            event_to_label = {int(frame): label for label, frame in enumerate(event_frames)}

            images, labels = [], []
            vid_name = '{}.mp4'.format(a['id'])
            vid_path = None
            for cand in self.vid_dir_candidates:
                cand_path = osp.join(cand, vid_name)
                if osp.exists(cand_path):
                    vid_path = cand_path
                    if self._resolved_vid_dir is None:
                        self._resolved_vid_dir = cand
                        self._log('using video directory: {}'.format(cand))
                    break
            if vid_path is None:
                vid_path = osp.join(self.vid_dir, vid_name)
            # self._log('start idx={} video={}'.format(idx, vid_path))
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                raise RuntimeError('Failed to open video: {}'.format(vid_path))

            if self.train:
                # random starting position, sample 'seq_length' frames
                start_frame = np.random.randint(events[-1] + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                pos = start_frame
                while len(images) < self.seq_length:
                    ret, img = cap.read()
                    if ret:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(event_to_label.get(pos, 8))
                        pos += 1
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        pos = 0
            else:
                # full clip
                for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                    _, img = cap.read()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(event_to_label.get(pos, 8))

            sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
            if self.transform:
                sample = self.transform(sample)
            # self._log('done idx={} frames={} elapsed={:.3f}s'.format(idx, len(images), time.time() - t0))
            return sample
        except Exception as e:
            self._log('error idx={} elapsed={:.3f}s err={}'.format(idx, time.time() - t0, repr(e)))
            raise
        finally:
            if cap is not None:
                cap.release()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


class RandomAugment(object):
    """
    Sequence-consistent augmentations for golf swing clips.
    The same geometric transform is applied to every frame in a sample.
    """
    def __init__(self,
                 p_flip=0.5,
                 max_rotate=8.0,
                 max_translate=0.05,
                 min_scale=0.95,
                 max_scale=1.05,
                 max_brightness=0.12,
                 max_contrast=0.12):
        self.p_flip = p_flip
        self.max_rotate = max_rotate
        self.max_translate = max_translate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_brightness = max_brightness
        self.max_contrast = max_contrast

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        if len(images) == 0:
            return sample

        h, w = images.shape[1], images.shape[2]
        do_flip = np.random.rand() < self.p_flip
        angle = np.random.uniform(-self.max_rotate, self.max_rotate)
        scale = np.random.uniform(self.min_scale, self.max_scale)
        tx = np.random.uniform(-self.max_translate, self.max_translate) * w
        ty = np.random.uniform(-self.max_translate, self.max_translate) * h
        contrast = 1.0 + np.random.uniform(-self.max_contrast, self.max_contrast)
        brightness = np.random.uniform(-self.max_brightness, self.max_brightness) * 255.0

        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        m[0, 2] += tx
        m[1, 2] += ty

        out = np.empty_like(images)
        for i in range(len(images)):
            frame = images[i]
            frame = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if do_flip:
                frame = np.ascontiguousarray(frame[:, ::-1, :])
            frame = frame.astype(np.float32) * contrast + brightness
            out[i] = np.clip(frame, 0, 255).astype(np.uint8)

        return {'images': out, 'labels': labels}


if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=64,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))
