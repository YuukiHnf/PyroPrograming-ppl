#一部GPUをしたく、GoogleColaboratoryで実行したものを写しています。pyro等のインストールは!pip install pyro-pplです
import torch
import numpy as np
import matplotlib.pyplot as plt
!pip install pyro-ppl
plt.style.use("ggplot")
import pyro
import pyro.infer as infer
import pyro.distributions as dist
import pyro.nn as nn
from torch.distributions import constraints

class NMF(pyro.nn.PyroModule):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.N = output_dim
    self.D = input_dim
    self.M = hidden_dim
    self.aW = pyro.nn.PyroParam(torch.tensor(1.0), constraint=constraints.positive)
    self.bW = pyro.nn.PyroParam(torch.tensor(1.0), constraint=constraints.positive)
    self.aH = pyro.nn.PyroParam(torch.tensor(1.0), constraint=constraints.positive)
    self.bH = pyro.nn.PyroParam(torch.tensor(1.0), constraint=constraints.positive)
    self.W = pyro.nn.PyroSample(lambda self: dist.Gamma(self.aW, self.bW).expand([self.D, self.M]).to_event(2))
    self.H = pyro.nn.PyroSample(lambda self: dist.Gamma(self.aH, self.bH).expand([self.M, self.N]).to_event(2))
    self.d_axis = pyro.plate("d_axis", self.D, dim=-2)
    self.n_axis = pyro.plate("n_axis", self.N, dim=-1)

  def forward(self, X=None):
    S = pyro.sample("S", dist.Poisson(self.W @ self.H).to_event(2), obs=X)
    return S



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Get Data
X = np.load('A.npy')
Train_data = torch.tensor(X, dtype=torch.float32, device=device)
print(Train_data.shape)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#train
model = NMF(Train_data.shape[0], 10, Train_data.shape[1])
nuts = infer.NUTS(model)

mcmc = infer.MCMC(nuts, num_samples=2500, warmup_steps=200)
mcmc.run(Train_data)

print(model.W)
print(model.H)
print(model.W @ model.H)
dell = (X - (W@H).detach().cpu().numpy())
print(np.linalg.norm(dell))
np.save("W_NMF_posson_K=10",W.detach().cpu().numpy())
np.save("H_NMF_posson_K=10", H.detach().cpu().numpy())
