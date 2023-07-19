import torch
from torch import nn

class edcorrector(nn.Module):
    def __init__(self, encoder, decoder, lamb):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lamb = lamb

    def forward(self, x):
        """
        INPUT
        x : image data
        OUTPUT
        z : modified latent data
        """
        
        
        """ toy example : same as random noise
        self.fixed_x = x
        z = torch.randn_like(self.encoder(x))
        z = minimize(self.loss, z, method='bfgs')
        return z.x
        """

        # return self.encoder(x) # induces same result, will use this as framework

    def loss(self, z):
        return torch.norm((self.encoder(self.decoder(z)-self.fixed_x))**2 + self.lamb * (self.encoder(self.fixed_x) - z)**2)