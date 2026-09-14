"""LightningModule for VolumetricSMPL training.

All data generation happens on GPU inside the step. The dataset therefore yields nothing
but integer indices: the pose cache is small enough (~185 MB) to live on the device, so
there is no host->device transfer, no DataLoader workers, and no /dev/shm pressure.

This also removes a redundancy in COAP, which ran the SMPL-X forward pass twice per
sample — once on CPU in `__getitem__` to generate points, once on GPU in the training
step to compute the loss.
"""
from __future__ import annotations

import pytorch_lightning as pl
import smplx
import torch

from VolumetricSMPL import attach_volume
from VolumetricSMPL.volumetric_smpl import VolumetricSMPL as _VS

from .amass import NUM_BETAS
from . import ragged
from .occupancy import occupancy_kaolin
from .sampling import sample_query_points

# Recovered from the released checkpoint's optimizer/scheduler state, not the paper.
# The paper says lr 1e-4; the checkpoint says 5e-4. Do not "correct" these back to the
# published text — see FINDINGS.md, "Recovered training recipe".
DEFAULTS = dict(
    lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    t_max=450_000, eta_min=1e-5,
    n_points=512, sigma=0.01, n_unif_samples=512,
    # The ragged in-box filter: 1.66x faster per step with bit-identical losses.
    # False falls back to the package's own `query_training`. See training/ragged.py.
    ragged_filter=True,
)


class IndexDataset(torch.utils.data.Dataset):
    """Yields row indices into the pose cache. Batching is all that is needed here."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


class VolumetricSMPLModule(pl.LightningModule):
    def __init__(self, train_params: dict, val_params: dict,
                 model_path: str = 'data/body_models', gender: str = 'neutral',
                 batch_size: int = 8, num_betas: int = NUM_BETAS, **cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['train_params', 'val_params'])
        self.cfg = {**DEFAULTS, **cfg}
        self.batch_size = batch_size

        # batch_size=1, NOT the training batch size. smplx 0.1.28's forward mixes the
        # runtime batch with the construction-time one:
        #   lmk_faces_idx  = self.lmk_faces_idx.unsqueeze(0).expand(batch_size, -1)
        #   lmk_bary_coords = self.lmk_bary_coords.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # so any batch smaller than the constructed size dies in vertices2landmarks with
        # "einsum(): subscript b has size N ... does not broadcast". Validation's final
        # batch is 316 % 8 = 4, which hits it every epoch. Building with 1 makes every
        # default buffer broadcast instead.
        body = smplx.create(model_path=model_path, model_type='smplx', gender=gender,
                            num_betas=num_betas, use_pca=False, flat_hand_mean=False,
                            batch_size=1)
        # pretrained=False: we are training these weights from scratch.
        self.body = attach_volume(body, pretrained=False)
        self.faces = torch.from_numpy(self.body.faces.astype('int64'))

        self._train_params = train_params
        self._val_params = val_params
        self.n_train = len(next(iter(train_params.values())))
        self.n_val = len(next(iter(val_params.values())))

    # --- plumbing ---------------------------------------------------------------
    def setup(self, stage=None):
        dev = self.device
        self._train_params = {k: v.to(dev) for k, v in self._train_params.items()}
        self._val_params = {k: v.to(dev) for k, v in self._val_params.items()}
        self.faces = self.faces.to(dev)

    def state_dict(self, *a, **kw):
        return self.body.volume.state_dict(*a, **kw)

    def load_state_dict(self, *a, **kw):
        return self.body.volume.load_state_dict(*a, **kw)

    def configure_optimizers(self):
        c = self.cfg
        opt = torch.optim.Adam(self.body.volume.parameters(), lr=c['lr'],
                               betas=c['betas'], eps=c['eps'],
                               weight_decay=c['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=c['t_max'], eta_min=c['eta_min'])
        # Per-step, not per-epoch: T_max is 450k *steps*. Training deliberately overruns
        # it (the original ran 471,644), so the cosine turns back up at the end.
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            IndexDataset(self.n_train), batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            IndexDataset(self.n_val), batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=0)

    # --- data generation --------------------------------------------------------
    @torch.no_grad()
    def _forward_bodies(self, params: dict, idx: torch.Tensor):
        """SMPL-X forward. Under no_grad: the body parameters are data, not learnables —
        only `volume`'s weights are trained, and gradients reach the PointNet encoder
        through its own layers rather than through the vertices."""
        kw = {k: v[idx] for k, v in params.items()}
        kw['expression'] = torch.zeros(len(idx), self.body.num_expression_coeffs,
                                       device=self.device)
        return self.body(**kw, return_verts=True, return_full_pose=True)

    @torch.no_grad()
    def _generate(self, smpl_output, n_points: int):
        """Query points and ground-truth occupancy, entirely on GPU."""
        pts = sample_query_points(self.body.volume, smpl_output, n_points,
                                  sigma=self.cfg['sigma'])
        B, K = pts.shape[0], pts.shape[1]
        flat = pts.reshape(B, K * n_points, 3)
        gt_occ = occupancy_kaolin(smpl_output.vertices, self.faces, flat)
        return flat, gt_occ, K

    # --- steps ------------------------------------------------------------------
    def training_step(self, idx, batch_idx):
        out = self._forward_bodies(self._train_params, idx)
        points, gt_occ, _ = self._generate(out, self.cfg['n_points'])

        if self.cfg['ragged_filter']:
            loss = ragged.query_training(
                self.body.volume, points, gt_occ, out,
                n_unif_samples=self.cfg['n_unif_samples'])
        else:
            loss = self.body.volume.query_training(
                points, gt_occ, out, n_unif_samples=self.cfg['n_unif_samples'])
        for key, val in loss.items():
            self.log(f'train/{key}', val, prog_bar=(key == 'total_loss'))
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss['total_loss']

    def on_train_batch_end(self, *a):
        # The impl_code cache is keyed on pose; dropping it each step prevents a stale
        # autograd graph being reused when a later batch happens to match.
        self.body.volume.detach_cache()

    @torch.no_grad()
    def validation_step(self, idx, batch_idx):
        out = self._forward_bodies(self._val_params, idx)
        n = self.cfg['n_points']
        points, gt_occ, K = self._generate(out, n)
        B = points.shape[0]

        pred = self.body.volume.query_occupancy(points, out)
        n_u = n // 2
        g, p = gt_occ.reshape(B, K, n), pred.reshape(B, K, n)
        iou_u = _VS.compute_iou(p[..., :n_u].reshape(B, -1), g[..., :n_u].reshape(B, -1))
        iou_s = _VS.compute_iou(p[..., n_u:].reshape(B, -1), g[..., n_u:].reshape(B, -1))
        self.body.volume.detach_cache()

        for key, val in (('iou_unif', iou_u), ('iou_surf', iou_s),
                         ('iou_mean', (iou_u + iou_s) * 0.5)):
            self.log(f'val/{key}', val * 100.0, on_epoch=True, sync_dist=True,
                     prog_bar=(key == 'iou_mean'))
