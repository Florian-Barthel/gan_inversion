import numpy as np
import torch
from lpips import lpips
from losses.id_loss import IDLoss


class Inversion:
    def __init__(self):
        self.device = torch.device("cuda")
        self.generator = None
        self.images = None
        self.cams = None
        self.optimizer = None
        self.loss_L2 = torch.nn.MSELoss(reduction="mean").to(self.device)
        self.loss_percept = lpips.LPIPS(net="vgg").to(self.device)
        self.ID_loss = IDLoss()

    def set_targets(self, images, cams):
        self.images = torch.tensor(images, device=self.device, dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0
        self.cams = torch.tensor(cams, device=self.device, dtype=torch.float32)

    def set_generator(self, generator):
        self.generator = generator
        w_samples = self.generator.mapping(torch.randn(100_000, 512, device=self.device), c=torch.zeros(100_000, 25, device=self.device))
        self.w_opt = torch.mean(w_samples, dim=0, keepdim=True).clone().detach().requires_grad_(True)
        self.optimizer_w = torch.optim.Adam(params=[self.w_opt], lr=0.0)
        self.optimizer_pti = torch.optim.Adam(params=self.generator.parameters(), lr=0.0)

    def calc_loss(self, gen_image, gt_image, weights):
        l2 = self.loss_L2(gen_image, gt_image)
        lpips = torch.mean(self.loss_percept(gen_image, gt_image))
        id = self.ID_loss(gen_image, gt_image)
        full = l2 * weights["mse_loss"] + lpips * weights["lpips_loss"] + id * weights["id_loss"]
        return full

    def _step(self, hyperparams, optimizer):
        self.optimizer_w.param_groups[0]["lr"] = hyperparams["lr"]
        batch_size = int(hyperparams["batch_size"])
        optimizer.zero_grad()
        select_indices = np.arange(len(self.images), dtype=int)
        if batch_size != len(self.images):
            select_indices = np.random.choice(select_indices, batch_size)
        ground_truth = self.images[select_indices]
        cam = self.cams[select_indices]
        generated_images = self.generator.synthesis(ws=self.w_opt.tile(len(self.images), 1), c=cam, random_bg=False)["image"]
        loss = self.calc_loss(generated_images, ground_truth, hyperparams)
        loss.backward()
        optimizer.step()
        return loss

    def step_w(self, hyperparams):
        loss = self._step(hyperparams, self.optimizer_w)
        return self.w_opt, loss

    def step_pti(self, hyperparams):
        loss = self._step(hyperparams, self.optimizer_pti)
        return self.w_opt, loss