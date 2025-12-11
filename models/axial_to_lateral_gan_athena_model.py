# TODO August 06 version (anisotropic-safe, per-axis slicing fixed, LSGAN default)
# MODIFIED: Added Lateral Identity Loss (idt_A_xy)
import torch
import itertools
import numpy as np
from util.image_pool import ImagePool  # kept for compatibility (not used)
from .base_model import BaseModel
from . import networks


class AxialToLateralGANAthenaModel(BaseModel):
    """
    3D Generator (A->B, B->A), 2D Discriminators on planes (XY, XZ, YZ).
    Works with anisotropic volumes. Input shape is [N, C, D, H, W] (Z,Y,X = D,H,W).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # Match paper defaults more closely
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='weight for forward cycle loss (A->B->A)')
            # --- NEW ARGUMENT: Identity Loss Weight ---
            parser.add_argument('--lambda_idt_xy', type=float, default=0.5,
                                help='weight for identity loss on XY plane (G_A(A) vs A). Prevents XY artifacts.')
            # ------------------------------------------
            parser.add_argument('--pool_size', type=int, default=50,
                                help='(unused) image buffer; kept for CLI compatibility')
            parser.add_argument('--gan_mode', type=str, default='lsgan',
                                help='GAN objective: [vanilla|lsgan|wgangp] (paper uses LSGAN)')
        # plane conversion (source plane => target plane) e.g. yz xy
        parser.add_argument('--conversion_plane', type=str, nargs='+', default=['yz', 'xy'],
                            help='Source plane then target plane. Example: yz xy')
        # IMPORTANT: order is [source, target, reference]
        parser.add_argument('--lambda_plane', type=int, nargs='+', default=[1, 1, 1],
                            help='weights [source, target, reference] for plane-wise G_A adversarial')
        parser.add_argument('--netG_B', type=str, default='deep_linear_gen',
                            help='generator for B->A path')
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        # ---- bookkeeping / loss names ----
        self.loss_names = [
            'D_A_xy', 'D_A_xz', 'D_A_yz',
            'G_A', 'G_A_xy', 'G_A_xz', 'G_A_yz',
            'cycle_A',
            'idt_A_xy',  # <--- Added to log this loss
            'D_B_xy', 'D_B_xz', 'D_B_yz',
            'G_B', 'G_B_xy', 'G_B_xz', 'G_B_yz'
        ]
        # In test mode, TestOptions doesn't define gan_mode, so default to LSGAN
        self.gan_mode = getattr(opt, "gan_mode", "lsgan")

        self.gen_dimension = 3  # 3D convs
        self.dis_dimension = 2  # 2D convs

        # visuals
        self.visual_names = ['real', 'fake', 'rec']

        # plane -> slice axis mapping
        plane_to_slice_axis = {'xy': 0, 'xz': 1, 'yz': 2}
        src_plane, tgt_plane = opt.conversion_plane[0], opt.conversion_plane[1]
        rem_plane = [p for p in plane_to_slice_axis if p not in (src_plane, tgt_plane)][0]

        print(f"source plane: {src_plane}")
        print(f"target plane: {tgt_plane}")
        print(f"remaining plane: {rem_plane}")

        self.source_sl_axis = plane_to_slice_axis[src_plane]
        self.target_sl_axis = plane_to_slice_axis[tgt_plane]
        self.remain_sl_axis = plane_to_slice_axis[rem_plane]

        # plane weights: order in CLI is [source, target, reference]
        src_w, tgt_w, ref_w = opt.lambda_plane
        denom = max(src_w + tgt_w + ref_w, 1e-8)
        self.lambda_plane_source = src_w / denom
        self.lambda_plane_target = tgt_w / denom
        self.lambda_plane_ref    = ref_w / denom

        # ---- models ----
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A_xy', 'D_A_xz', 'D_A_yz', 'D_B_xy', 'D_B_xz', 'D_B_yz']
        else:
            self.model_names = ['G_A', 'G_B']

        # Generators (3D)
        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dimension=self.gen_dimension
        )
        self.netG_B = networks.define_G(
            opt.output_nc, opt.input_nc, opt.ngf, opt.netG_B, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dimension=self.gen_dimension
        )

        if self.isTrain:
            # Discriminators (2D)
            self.netD_A_yz = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension
            )
            self.netD_A_xy = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension
            )
            self.netD_A_xz = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension
            )

            self.netD_B_yz = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension
            )
            self.netD_B_xy = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension
            )
            self.netD_B_xz = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, False, self.gpu_ids, dimension=self.dis_dimension
            )

            # losses / opts
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            # We use criterionCycle (L1) for identity loss as well
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(
                    self.netD_A_yz.parameters(), self.netD_A_xy.parameters(), self.netD_A_xz.parameters(),
                    self.netD_B_yz.parameters(), self.netD_B_xy.parameters(), self.netD_B_xz.parameters()
                ),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # --------------------
    # I/O and forward pass
    # --------------------
    def set_input(self, input):
        """Expect dict with keys 'A'/'B' and 'A_paths'/'B_paths'."""
        AtoB = self.opt.direction == 'AtoB'
        x = input['A' if AtoB else 'B'].to(self.device)  # [N,C,D,H,W] or [C,D,H,W]
        paths = input['A_paths' if AtoB else 'B_paths']

        if x.dim() == 4:  # [C,D,H,W] -> add batch
            x = x.unsqueeze(0)
        assert x.dim() == 5, f"Expected 5D [N,C,D,H,W], got {list(x.shape)}"

        self.real = x
        self.image_paths = paths

        # record sizes (Z,Y,X) = (D,H,W)
        N, C, D, H, W = self.real.shape
        self.depth_D, self.height_H, self.width_W = D, H, W

    def forward(self):
        """A->B->A pass."""
        self.fake = self.netG_A(self.real)   # A -> B
        self.rec  = self.netG_B(self.fake)   # B -> A

    # --------------------
    # Discriminator helpers
    # --------------------
    def _num_slices_for_axis(self, vol, slice_axis:int):
        """Return #slices to iterate for the given axis (0=D, 1=H, 2=W)."""
        assert vol.dim() == 5, "vol must be [N,C,D,H,W]"
        if slice_axis == 0:   # XY plane => vary along D (Z)
            return vol.shape[2]
        elif slice_axis == 1: # XZ plane => vary along H (Y)
            return vol.shape[3]
        elif slice_axis == 2: # YZ plane => vary along W (X)
            return vol.shape[4]
        else:
            raise ValueError("slice_axis must be 0,1,2")

    def _get_slice(self, vol, slice_axis:int, idx:int):
        """Return [N,C,H,W] slice from [N,C,D,H,W] along axis."""
        if slice_axis == 0:   # XY @ depth idx
            return vol[:, :, idx, :, :]
        elif slice_axis == 1: # XZ @ row idx
            return vol[:, :, :, idx, :]
        elif slice_axis == 2: # YZ @ col idx
            return vol[:, :, :, :, idx]
        else:
            raise ValueError("slice_axis must be 0,1,2")

    def iter_f(self, vol, disc, slice_axis:int):
        """
        Apply 2D discriminator across all slices of vol along slice_axis,
        stack outputs along a new "depth" dim to return [N, C_out, S, H_out, W_out].
        """
        # probe output shape
        first_slice = self._get_slice(vol, slice_axis, 0)       # [N,C,H,W]
        probe = disc(first_slice)                                # [N,Cd,Hd,Wd]
        N, Cd, Hd, Wd = probe.shape

        S = self._num_slices_for_axis(vol, slice_axis)           # number of slices along this axis
        out = probe.new_zeros((N, Cd, S, Hd, Wd))                # allocate

        # fill
        for i in range(S):
            out[:, :, i, :, :] = disc(self._get_slice(vol, slice_axis, i))

        return out

    # --------------------
    # Backward / losses
    # --------------------
    def backward_D_basic(self, netD, real, fake, slice_axis_real, slice_axis_fake):
        """Standard GAN D loss using per-slice discriminator passes."""
        pred_real = self.iter_f(real, netD, slice_axis_real)
        pred_fake = self.iter_f(fake.detach(), netD, slice_axis_fake)

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        return loss_D

    # D_A: compare real A vs fake B per plane
    def backward_D_A_xy(self):
        self.loss_D_A_xy = self.backward_D_basic(self.netD_A_xy, self.real, self.fake,
                                                 self.target_sl_axis, self.target_sl_axis)

    def backward_D_A_yz(self):
        self.loss_D_A_yz = self.backward_D_basic(self.netD_A_yz, self.real, self.fake,
                                                 self.target_sl_axis, self.source_sl_axis)

    def backward_D_A_xz(self):
        self.loss_D_A_xz = self.backward_D_basic(self.netD_A_xz, self.real, self.fake,
                                                 self.target_sl_axis, self.remain_sl_axis)

    # D_B: compare real A vs reconstructed A per plane
    def backward_D_B_xy(self):
        self.loss_D_B_xy = self.backward_D_basic(self.netD_B_xy, self.real, self.rec,
                                                 self.target_sl_axis, self.target_sl_axis)

    def backward_D_B_yz(self):
        self.loss_D_B_yz = self.backward_D_basic(self.netD_B_yz, self.real, self.rec,
                                                 self.source_sl_axis, self.source_sl_axis)

    def backward_D_B_xz(self):
        self.loss_D_B_xz = self.backward_D_basic(self.netD_B_xz, self.real, self.rec,
                                                 self.remain_sl_axis, self.remain_sl_axis)

    def backward_G(self):
        """Generator losses: plane-wise adversarial + cycle (A->B->A)."""
        lambda_A = self.opt.lambda_A

        # A->B plane-wise adversarial (weighted)
        self.loss_G_A_xy = self.criterionGAN(self.iter_f(self.fake, self.netD_A_xy, self.target_sl_axis), True) * self.lambda_plane_target
        self.loss_G_A_yz = self.criterionGAN(self.iter_f(self.fake, self.netD_A_yz, self.source_sl_axis), True) * self.lambda_plane_source
        self.loss_G_A_xz = self.criterionGAN(self.iter_f(self.fake, self.netD_A_xz, self.remain_sl_axis), True) * self.lambda_plane_ref
        self.loss_G_A = self.loss_G_A_xy + self.loss_G_A_yz + self.loss_G_A_xz

        # B->A plane-wise adversarial (uniform thirds)
        self.loss_G_B_xy = self.criterionGAN(self.iter_f(self.rec, self.netD_B_xy, self.target_sl_axis), True) * (1/3)
        self.loss_G_B_yz = self.criterionGAN(self.iter_f(self.rec, self.netD_B_yz, self.source_sl_axis), True) * (1/3)
        self.loss_G_B_xz = self.criterionGAN(self.iter_f(self.rec, self.netD_B_xz, self.remain_sl_axis), True) * (1/3)
        self.loss_G_B = self.loss_G_B_xy + self.loss_G_B_yz + self.loss_G_B_xz

        # forward cycle only: || G_B(G_A(A)) - A ||
        self.loss_cycle_A = self.criterionCycle(self.rec, self.real) * lambda_A

        # --- NEW LOSS: Lateral Identity (XY preservation) ---
        # G_A(A) should ~ A in the lateral plane to avoid hallucinations.
        # Since A is [N,C,D,H,W] and D is the stack of XY planes, L1(A, G(A)) works.
        if self.opt.lambda_idt_xy > 0:
            self.loss_idt_A_xy = self.criterionIdt(self.fake, self.real) * self.opt.lambda_idt_xy
        else:
            self.loss_idt_A_xy = 0
        # ----------------------------------------------------

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_idt_A_xy
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # G step
        self.set_requires_grad([self.netD_A_xy, self.netD_A_yz, self.netD_A_xz,
                                self.netD_B_xy, self.netD_B_yz, self.netD_B_xz], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D step
        self.set_requires_grad([self.netD_A_xy, self.netD_A_yz, self.netD_A_xz,
                                self.netD_B_xy, self.netD_B_yz, self.netD_B_xz], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A_xy()
        self.backward_D_A_yz()
        self.backward_D_A_xz()
        self.backward_D_B_xy()
        self.backward_D_B_yz()
        self.backward_D_B_xz()
        self.optimizer_D.step()