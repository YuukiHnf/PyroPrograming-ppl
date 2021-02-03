#一部GPUをしたく、GoogleColaboratoryで実行したものを写しています。pyro等のインストールは!pip install pyro-pplです
import torch
import numpy as np
import matplotlib.pyplot as plt
!pip install pyro-ppl
plt.style.use("ggplot")
import pyro.distributions as dist
import torch.nn as nn
import pyro

class LM(pyro.nn.PyroModule):
  def __init__(self, data, hidden_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.data = data
    self.data_num = data.shape[0]
    self.output_dim = self.data.shape[1]
    self.cuda()
    self.W = pyro.nn.PyroSample(dist.Normal(0.,10.).expand([self.output_dim, self.hidden_dim]).to_event(2))
    self.b = pyro.nn.PyroSample(dist.LogNormal(0.,10.).expand([self.output_dim]).to_event(1))
    #self.X = pyro.nn.PyroSample(dist.Normal(0.,1.).expand([self.hidden_dim]))
    self.scale = pyro.nn.PyroParam(torch.tensor(1.0),constraint=torch.distributions.constraints.positive)
  def forward(self, Y=None):
    if Y is None:
      Y = self.data
    W = self.W
    b = self.b
    #print(self.data_num)
    with pyro.plate("plate_data", self.data_num):
      x = pyro.sample("x",dist.MultivariateNormal(torch.zeros(self.hidden_dim),torch.eye(self.hidden_dim)))
      mean = (x@W.T) + b.reshape(1,-1)
      #return pyro.sample("obs", dist.MultivariateNormal(mean, 10*torch.eye(self.output_dim)), obs=Y)
      for i in pyro.plate("data_loop", self.output_dim):
        return pyro.sample("obs", dist.Normal(mean[..., i],self.scale), obs=Y[...,i])



input_dim = 2
#set
X = np.load('A.npy')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.tensor(X, dtype=torch.float32, device=device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
pyro.clear_param_store()
#Train
model = LM(data, 2)
nuts_kernel = pyro.infer.NUTS(model)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(data)
#plot
W = mcmc.get_samples()["W"]
b = mcmc.get_samples()["b"]
X = mcmc.get_samples()["x"]
W_posterior = W.mean(0)
b_posterior = b.mean(0)
X_posterior = X.mean(0)
result = X_posterior.detach().cpu().numpy()
plt.title("Linear Dimension Reducation")
plt.scatter(result[:,0], result[:,1])
plt.savefig("twodimension.png")

#plotAgain
from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=10).fit(np.load('A.npy'))
plt.figure(figsize=(20,20))
plt.xlim(-0.1,0.15)
plt.ylim(-0.12,0.15)
plt.scatter(result[:,0], result[:,1], label=kmean.labels_)
for i, txt in enumerate(title):
  plt.annotate(txt[:9], (result[i,0], result[i,1]))
plt.title("Linear Dimension Reducation with kmeans10")
plt.savefig("twodimensionWithKmeans.png")
