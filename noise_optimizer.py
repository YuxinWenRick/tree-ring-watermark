import torch
from torch import nn

class edcorrector():
    def __init__(self, encoder, decoder, lamb, num_iters=100, lr=0.001):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lamb = lamb
        self.unet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, stride=2),
        )
        self.num_iters = num_iters
        self.lr = lr

    def __call__(self, x):
        """
        INPUT
        x : image data (1, 3, 512, 512)
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z) and norm(d(z)-x)

        """
        z = torch.randn_like(self.encoder(x))
        z.requires_grad_(True)

        optimizer = torch.optim.SGD([z], lr=self.lr)
        for i in range(self.num_iters):
            z_pred = self.unet(z)
            loss = self.Customloss(x, z)
            print(f"Iteration {i}, Loss: {loss.item():.3f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return z
    
    def Customloss(self, x, z):
        loss1 = torch.mean((self.encoder(x)-z)**2)
        loss2 = self.lamb * torch.mean((self.decoder(z)-x)**2)
        loss = loss1 + loss2
        loss = torch.Tensor(loss, requires_grad=True)
        return loss

        """ toy example : just returning 
        z = self.encoder(x)
        return z
        """
        """ toy example : same as random noise
        self.fixed_x = x
        z = torch.randn_like(self.encoder(x))
        z = minimize(self.loss, z, method='bfgs')
        return z.x

    def loss(self, z):
        return torch.norm((self.encoder(self.decoder(z)-self.fixed_x))**2 + self.lamb * (self.encoder(self.fixed_x) - z)**2)
        """

        # return self.encoder(x) # induces same result, will use this as framework