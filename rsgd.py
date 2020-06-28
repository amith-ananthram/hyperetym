from torch.optim.optimizer import Optimizer

class RSGD(Optimizer):
    def __init__(self, params, lr):
        self.params = params
        self.manifold = manifold
        self.lr = lr

    def step(self, lr=None):

        loss = None
        
        # param_groups from Optimizer
        for group in self.param_groups:
            for p in group['param']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']

                dp = p.grad.data
                if dp.is_sparse:
                    p_data = dp.coalesce()

                dp = self.manifold.rgrad(p.data, dp)
                dp.mul_(-self.lr)

                self.manifold.expm(p.data, dp)

        return loss
