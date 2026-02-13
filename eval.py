from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import os


def eval(model, split, seq_length, n_cpu, disp, device=None, use_amp=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             pin_memory=device.type == 'cuda',
                             drop_last=False)

    correct = []

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            images, labels = sample['images'], sample['labels']
            probs_batches = []
            # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                image_batch = image_batch.to(device, non_blocking=device.type == 'cuda')
                with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                    logits = model(image_batch)
                probs_batches.append(F.softmax(logits, dim=1).cpu().numpy())
                batch += 1
            probs = np.concatenate(probs_batches, axis=0)
            _, _, _, _, c = correct_preds(probs, labels.squeeze())
            if disp:
                print(i, c)
            correct.append(c)
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    split = 1
    seq_length = 64
    n_cpu = 6

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    ckpt_path = os.getenv('GOLFDB_EVAL_CKPT', 'models/swingnet_best.pth.tar')
    if not os.path.exists(ckpt_path):
        ckpt_path = 'models/swingnet_1800.pth.tar'
    try:
        save_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only arg.
        save_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(save_dict['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True, device=device, use_amp=device.type == 'cuda')
    print('Checkpoint: {}'.format(ckpt_path))
    print('Average PCE: {}'.format(PCE))
