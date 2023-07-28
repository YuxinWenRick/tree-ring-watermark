import torch
from torch import nn

def softThreshold(x, l):
    return torch.sign(x) * torch.maximum(torch.abs(x)-l, torch.zeros(x.shape).to(x.device))

class edcorrector():
    def __init__(self, encoder, decoder, lamb, num_iters=300, lr=0.01):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
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
        z = self.encoder(x) # initial z
        xf = x.float()
        print(f"\titeration : start, norm : {torch.norm(self.decoder(z.half())-xf)}")
        t = self.lr
        for i in range(self.num_iters):
            z = softThreshold(z + t * self.encoder(x - self.decoder(z.half())), self.lamb)
            if i%1==0:
                print(f"\titeration : {i}, norm : {torch.norm(self.decoder(z.half())-xf)}")
        return z
        """
        z = self.encoder(x)
        for i in range(self.num_iters):
            if i%10==0:
                print(f"iteration : {i}, norm : {torch.norm(self.decoder(z)-x)}")
            grad = self.encoder(self.decoder(z)-x)
            z = z - self.lr * grad

        return z
        """

# class edcorrector():
#     def __init__(self, encoder, decoder, lamb, num_iters=100, lr=0.001):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.lamb = lamb
#         self.net = simplenet()
#         self.num_iters = num_iters
#         self.lr = lr

#     @torch.enable_grad()
#     def __call__(self, x):
#         """
#         INPUT
#         x : image data (1, 3, 512, 512)
#         OUTPUT
#         z : modified latent data (1, 4, 64, 64)

#         Goal : minimize norm(e(x)-z) and norm(d(z)-x)

#         """
#         z_pred = torch.randn_like(self.encoder(x))

#         self.net = self.net.float()
#         self.net.to(x.device)
#         self.net.train()
#         x = x.float()
#         z_pred = z_pred.float()
#         x.requires_grad_(True)
#         z_pred.requires_grad_(True)

#         optimizer = torch.optim.SGD([z_pred], lr=self.lr)
#         loss_func = torch.nn.MSELoss(reduction='sum')

#         for i in range(self.num_iters):
#             output = self.net(self.decoder(z_pred.half()).float())
#             #loss = self.Customloss(z_pred, x)
#             loss = loss_func(output, x)
#             print(f"Iteration {i}, Loss: {loss.item():.3f}")
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         return z_pred.half()
    
#     def Customloss(self, z, x):
#         loss1 = torch.nn.MSELoss()(self.encoder(x.half()), z)
#         loss2 = self.lamb * torch.nn.L1Loss()(self.decoder(z.half(), x))
#         loss = loss1 + loss2
#         return loss

#         """ toy example : just returning 
#         z = self.encoder(x)
#         return z
#         """
#         """ toy example : same as random noise
#         self.fixed_x = x
#         z = torch.randn_like(self.encoder(x))
#         z = minimize(self.loss, z, method='bfgs')
#         return z.x

#     def loss(self, z):
#         return torch.norm((self.encoder(self.decoder(z)-self.fixed_x))**2 + self.lamb * (self.encoder(self.fixed_x) - z)**2)
#         """

#         # return self.encoder(x) # induces same result, will use this as framework