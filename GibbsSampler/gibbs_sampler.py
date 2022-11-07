import random
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

'''
theta ~ Dirichlet, [K x 1]
Z ~ Cat(theta), idicator scaler
beta ~ Gaussian, [K x f]
X ~ Mix Gaussian, [N x f]
'''
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)


class GibbsSampler(object):
    def __init__(self) -> None:
        pass

    def sample_theta(self):
        raise NotImplementedError

    def sample_beta(self):
        raise NotImplementedError

    def sample_z(self):
        raise NotImplementedError


class GaussianGibbsSampler(GibbsSampler):

    def __init__(self, X, K=1):
        self.K = K
        self.X = X

    def init_theta(self):
        self.alpha = np.ones(self.K)
        return self.alpha / self.K
    
    def init_beta(self):
        n, _ = self.X.shape
        idx = np.random.choice(n, self.K, False)
        return self.X[idx]
    
    def init_z(self):
        n, _ = self.X.shape    
        return np.random.choice(self.K, n)
    
    def sample_theta(self, z):
        cnt = np.eye(self.K)[z].sum(axis=0)
        theta = st.dirichlet.rvs(self.alpha + cnt)
        return theta.flatten()
    
    def sample_beta(self, beta_mean, beta_var, z):
        
        n, f = np.shape(self.X)
        sigma2 = np.var(self.X, axis=0).mean()
        z_onehot = np.eye(self.K)[z]
        beta = np.zeros((self.K, f))
        for k in range(self.K):
            n_k = z_onehot.sum(axis=0)[k]
            x_k = self.X[z==k].mean(axis=0) # [1 * f]
            post_beta_var = 1 / (n_k / sigma2 + 1 / beta_var)
            post_beta_mean = (beta_mean / beta_var + n_k * x_k / sigma2) * post_beta_var
            post_beta_var = np.diag(post_beta_var)
            beta[k] = st.multivariate_normal.rvs(post_beta_mean, post_beta_var)
            
        return beta
    
    def sample_z(self, z, theta, beta):
        
        n, f = np.shape(self.X)
        sigma2 = np.var(self.X, axis=0).mean()
        weight = np.zeros((n, self.K))
        for k in range(self.K):
            mean_x = beta[k]
            var_x = np.diag(np.ones(f) * sigma2)
            weight[:, k] = st.multivariate_normal.pdf(self.X, mean_x, var_x)
        weight *= theta
        weight = weight / weight.sum(axis=1, keepdims=True) # TODO: prob NaN value
        z = [np.random.choice(self.K, p=weight[i]) for i in range(n)]
        return np.array(z)
    
    def log_joint(self, epoch, theta, beta, z, beta_mean, beta_var):
        n, f = np.shape(self.X)
        sigma2 = np.var(self.X, axis=0).mean()
        z_onehot = np.eye(self.K)[z]
        
        p_theta = np.sum(st.dirichlet.logpdf(theta, alpha=self.alpha))
        p_beta = np.sum([
            st.multivariate_normal.logpdf(beta[k], beta_mean, beta_var)
            for k in range(self.K)
        ])
        p_z = np.sum(st.multinomial.logpmf(z_onehot, 1, theta))
        p_x = 0
        for k in range(self.K):
            p_x += np.sum(st.multivariate_normal.logpdf(self.X[z==k], beta[k], np.diag(np.ones(f) * sigma2)))

        return pd.DataFrame({
            "log_joint": np.sum([p_theta, p_beta, p_z, p_x]),
            "logp_theta": p_theta,
            "logp_beta": p_beta,
            "logp_z": p_z,
            "logp_x": p_x,
        }, index = [epoch])
    
    
    def sampling(self, n_epochs=50):
        
        theta = self.init_theta()
        beta = self.init_beta()
        z = self.init_z()
        beta_mean_0 = np.zeros(self.X.shape[1]) - 10
        beta_var_0 = np.ones(self.X.shape[1])
        
        res = []
        for epoch in tqdm(range(n_epochs)):
            theta = self.sample_theta(z)
            beta = self.sample_beta(beta_mean_0, beta_var_0, z)
            z = self.sample_z(z, theta, beta)
            res.append(self.log_joint(epoch, theta, beta, z, beta_mean_0, beta_var_0))
        
        res = pd.concat(res)
        fig, axes = plt.subplots(3, 2, figsize=(8,6))
        res["log_joint"].plot(title="log joint probability", ax=axes[0,0])
        res["logp_theta"].plot(title="log p(theta)", ax=axes[0,1])
        res["logp_beta"].plot(title="log p(beta)", ax=axes[1,0])
        res["logp_z"].plot(title="log p(z)", ax=axes[1,1])
        res["logp_x"].plot(title="log p(x)", ax=axes[2,0])
        plt.tight_layout()
        
        return dict(res=res, z=z, theta=theta, beta=beta,)

    

if __name__ == "__main__":
    
    set_seed()
    gibbs = GaussianGibbsSampler(X=np.random.randn(1000, 2),K=3)
    res = gibbs.sampling()
    from IPython import display
    display(res["res"])

