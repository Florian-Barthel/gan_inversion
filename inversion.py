import numpy as np
import torch
from easydict import EasyDict
from lpips import lpips
from losses.id_loss import IDLoss


class Inversion:
    def __init__(self, generator, images, cams):
        self.device = torch.device("cuda")
        self.generator = generator
        self.images = images
        self.cams = cams
        self.w_opt = self.calc_average_w().clone().detach().requires_grad_(True)
        self.optimizer = torch.optim.Adam(params=[
            {"params": self.generator.parameters(), "lr": 0.0},
            {"params": self.w_opt, "lr": 0.0},
        ])

        self.loss_L2 = torch.nn.MSELoss(reduction="mean").to(self.device)
        self.loss_percept = lpips.LPIPS(net="vgg").to(self.device)
        self.ID_loss = IDLoss()

    def calc_average_w(self, num=100_000):
        w_samples = self.generator.mapping(torch.randn(num, 512, device=self.device))
        return torch.mean(w_samples, dim=0, keepdim=True)

    def update_lr(self, lr_gen, lr_w):
        self.optimizer.param_groups[0]["lr"] = lr_gen
        self.optimizer.param_groups[1]["lr"] = lr_w

    def calc_loss(self, gen_image, gt_image, weights):
        weights = EasyDict(weights)
        losses = EasyDict()
        losses.l2 = self.loss_L2(gen_image, gt_image)
        losses.lpips = torch.mean(self.loss_percept(gen_image, gt_image))
        losses.id = self.ID_loss(gen_image, gt_image)
        losses.full = losses.l2 * weights.mse_loss + losses.lpips * weights.lpips_loss + losses.id * weights.id_loss
        return losses

    def step(self, hyperparams, w_inversion: bool, pti: bool) -> dict:
        self.w_opt = self.w_opt.requires_grad_(w_inversion)
        self.generator = self.generator.requires_grad_(pti)
        self.update_lr(hyperparams["lr_w"], hyperparams["lr_pti"])
        batch_size = int(hyperparams["batch_size"])

        self.optimizer.zero_grad()
        select_indices = np.arange(len(self.images), dtype=int)
        if batch_size != len(self.images):
            select_indices = np.random.choice(select_indices, batch_size)
        ground_truth = self.images[select_indices]
        cam = self.cams[select_indices]
        generated_images = self.generator.synthesis(ws=self.w_opt.tile(len(self.images), 1), c=cam, random_bg=False)["image"]
        loss = self.calc_loss(generated_images, ground_truth, hyperparams)
        loss.full.backward()
        self.optimizer.step()
        return loss