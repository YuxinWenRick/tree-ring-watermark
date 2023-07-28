import copy

import torch
from torch import nn

class edcorrector():
    def __init__(self, pipe, lamb, num_iters=300, lr=0.01):
        super().__init__()
        self.pipe = pipe
        self.encoder = copy.deepcopy(self.pipe.get_image_latents)
        self.decoder = copy.deepcopy(self.pipe.decode_image)
        self.net = copy.deepcopy(self.pipe.unet)
        self.lamb = lamb
        self.num_iters = num_iters
        self.lr = lr

    @torch.enable_grad()
    def __call__(self, x):
        """
        INPUT
        x : image data (1, 3, 512, 512) -> given data
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z) and norm(d(z)-x)
        """
        z = self.encoder(x).clone() # initial z
        z.requires_grad_(True)
        loss_function = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD([z], lr=self.lr)

        for i in range(self.num_iters):
            out = self.net(x, 0, None)
            out = self.pipe.scheduler.convert_model_output(out, z)
            z_pred = self.pipe.scheduler.dpm_solver_first_order_update(out, z)
            loss = loss_function(z_pred, self.encoder(x))
            print(f"t: {t}, Iteration {i}, Loss: {loss.item():.3f}")
            if loss.item() < 0.001:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return z.half()