from torch.optim.optimizer import Optimizer

class RSGD(Optimizer):
	def __init__(self, params, manifold, lr):
		defaults = {
			'lr': lr,
		}
		super(RSGD, self).__init__(params, defaults)
		self.manifold = manifold
		self.lr = lr

	def step(self, lr=None):
		# param_groups from Optimizer
		for group in self.param_groups:
			for p in group['params']:
				lr = lr or group['lr']

				if p.grad is None:
					continue
					
				dp = p.grad.data
				if dp.is_sparse:
					dp = dp.coalesce()

				dp = self.manifold.rgrad(p.data, dp)
				dp.mul_(-lr)

				self.manifold.expm(p.data, dp)