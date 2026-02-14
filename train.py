from dataloader import GolfDB, Normalize, ToTensor, RandomAugment
from model import EventDetector
from util import *
from eval import eval as run_eval
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time


def _log(msg):
    print('[train][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


def _model_state_dict(model):
    # torch.compile wraps the module; save plain weights for compatibility with eval/test scripts.
    return model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs).pow(self.gamma)
        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, targets)
            focal_factor = focal_factor * alpha_factor
        loss = -focal_factor * target_log_probs
        return loss.mean()


if __name__ == '__main__':

    # training configuration
    split = 1
    iterations = int(os.getenv('GOLFDB_ITERATIONS', '2000'))
    it_save = int(os.getenv('GOLFDB_SAVE_INTERVAL', '100'))
    n_cpu = 6
    seq_length = int(os.getenv('GOLFDB_SEQ_LENGTH', '64'))
    bs = int(os.getenv('GOLFDB_BATCH_SIZE', '22'))
    k = int(os.getenv('GOLFDB_FREEZE_LAYERS', '0'))  # fewer frozen layers usually improve final accuracy
    debug_train = os.getenv('GOLFDB_DEBUG_TRAIN', '0') == '1'
    log_every = int(os.getenv('GOLFDB_LOG_EVERY', '10'))
    dataloader_timeout_s = int(os.getenv('GOLFDB_DATALOADER_TIMEOUT_S', '60'))
    num_workers = int(os.getenv('GOLFDB_NUM_WORKERS', str(n_cpu)))
    use_amp = os.getenv('GOLFDB_USE_AMP', '1') == '1'
    use_compile = os.getenv('GOLFDB_USE_COMPILE', '0') == '1'
    pin_memory = os.getenv('GOLFDB_PIN_MEMORY', '1') == '1'
    persistent_workers = os.getenv('GOLFDB_PERSISTENT_WORKERS', '1') == '1'
    prefetch_factor = int(os.getenv('GOLFDB_PREFETCH_FACTOR', '4'))
    max_grad_norm = float(os.getenv('GOLFDB_MAX_GRAD_NORM', '1.0'))
    grad_accum_steps = max(1, int(os.getenv('GOLFDB_GRAD_ACCUM_STEPS', '1')))
    lr = float(os.getenv('GOLFDB_LR', '0.001'))
    onecycle_pct_start = float(os.getenv('GOLFDB_ONECYCLE_PCT_START', '0.1'))
    loss_type = os.getenv('GOLFDB_LOSS', 'ce').lower()  # ce or focal
    focal_gamma = float(os.getenv('GOLFDB_FOCAL_GAMMA', '2.0'))
    label_smoothing = float(os.getenv('GOLFDB_LABEL_SMOOTHING', '0.0'))
    use_augment = os.getenv('GOLFDB_USE_AUGMENT', '1') == '1'
    aug_flip_p = float(os.getenv('GOLFDB_AUG_FLIP_P', '0.5'))
    aug_rotate = float(os.getenv('GOLFDB_AUG_ROTATE', '8.0'))
    aug_translate = float(os.getenv('GOLFDB_AUG_TRANSLATE', '0.05'))
    aug_scale_min = float(os.getenv('GOLFDB_AUG_SCALE_MIN', '0.95'))
    aug_scale_max = float(os.getenv('GOLFDB_AUG_SCALE_MAX', '1.05'))
    aug_brightness = float(os.getenv('GOLFDB_AUG_BRIGHTNESS', '0.12'))
    aug_contrast = float(os.getenv('GOLFDB_AUG_CONTRAST', '0.12'))
    eval_interval = int(os.getenv('GOLFDB_EVAL_INTERVAL', str(it_save)))
    eval_n_cpu = int(os.getenv('GOLFDB_EVAL_NUM_WORKERS', str(min(max(num_workers, 1), 4))))
    eval_disp = os.getenv('GOLFDB_EVAL_DISP', '0') == '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    if num_workers == 0:
        dataloader_timeout_s = 0
        persistent_workers = False

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    _log('frozen layers: {}'.format(k))
    model.train()
    model.to(device)
    if use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    train_tfms = []
    if use_augment:
        train_tfms.append(RandomAugment(p_flip=aug_flip_p,
                                        max_rotate=aug_rotate,
                                        max_translate=aug_translate,
                                        min_scale=aug_scale_min,
                                        max_scale=aug_scale_max,
                                        max_brightness=aug_brightness,
                                        max_contrast=aug_contrast))
    train_tfms.extend([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose(train_tfms),
                     train=True)
    _log('dataset ready: samples={}'.format(len(dataset)))
    _log('device={} amp={} pin_memory={}'.format(device.type, use_amp and device.type == 'cuda', pin_memory))
    _log('dataloader config: batch_size={} num_workers={} timeout_s={} persistent_workers={} prefetch_factor={} drop_last={}'.format(
        bs, num_workers, dataloader_timeout_s, persistent_workers, prefetch_factor if num_workers > 0 else 'n/a', True))
    _log('optimization config: lr={} loss={} label_smoothing={} grad_accum_steps={} augment={}'.format(
        lr, loss_type, label_smoothing, grad_accum_steps, use_augment))
    _log('eval config: interval={} eval_workers={}'.format(eval_interval, eval_n_cpu))

    loader_kwargs = {
        'batch_size': bs,
        'shuffle': True,
        'num_workers': num_workers,
        'timeout': dataloader_timeout_s,
        'drop_last': True,
        'pin_memory': pin_memory and device.type == 'cuda'
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor
    data_loader = DataLoader(dataset, **loader_kwargs)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).to(device)
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    total_steps=iterations,
                                                    pct_start=onecycle_pct_start)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')

    losses = AverageMeter()
    best_pce = -1.0

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    while i < iterations:
        last_batch_end = time.time()
        for batch_idx, sample in enumerate(data_loader):
            fetch_elapsed = time.time() - last_batch_end
            if debug_train:
                _log('batch {} fetched in {:.3f}s'.format(batch_idx, fetch_elapsed))
            images = sample['images'].to(device, non_blocking=pin_memory and device.type == 'cuda')
            labels = sample['labels'].to(device, non_blocking=pin_memory and device.type == 'cuda')
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                logits = model(images)
                labels = labels.view(bs * seq_length)
                loss = criterion(logits, labels)
            scaler.scale(loss / grad_accum_steps).backward()
            losses.update(loss.item(), images.size(0))
            should_step = ((micro_step + 1) % grad_accum_steps == 0)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                i += 1
                if i % log_every == 0 or i == 1:
                    print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})\tLR: {lr:.6f}'.format(
                        i, loss=losses, lr=scheduler.get_last_lr()[0]), flush=True)
                if i % it_save == 0:
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'model_state_dict': _model_state_dict(model),
                                'iteration': i}, 'models/swingnet_{}.pth.tar'.format(i))
                if eval_interval > 0 and i % eval_interval == 0:
                    model.eval()
                    pce = run_eval(model, split, seq_length, eval_n_cpu, eval_disp,
                                   device=device, use_amp=use_amp and device.type == 'cuda')
                    _log('validation pce at iter {}: {:.4f}'.format(i, pce))
                    if pce > best_pce:
                        best_pce = pce
                        torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                    'model_state_dict': _model_state_dict(model),
                                    'iteration': i,
                                    'best_pce': best_pce}, 'models/swingnet_best.pth.tar')
                        _log('new best checkpoint: models/swingnet_best.pth.tar (pce={:.4f})'.format(best_pce))
                    model.train()
                if i == iterations:
                    break
            last_batch_end = time.time()
            micro_step += 1
