from torch.optim.optimizer import Optimizer

class RSGD(Optimizer):

    def __init__(self, params, lr, rgrad, expm):

        self.params = params
        self.lr = lr
        self.rgrad = rgrad
        self.expm = expm

    def step(self, lr=None):

        loss = None
        
        #param_groups from Optimizer
        for group in self.param_groups:
            for p in group['param']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']

                dp = p.grad.data
                if dp.is_sparse:
                    p_data = dp.coalesce()

                dp = rgrad(p.data, dp)
                dp.mul_(-self.lr)

                self.expm(p.data, dp)

        return loss

        
    



