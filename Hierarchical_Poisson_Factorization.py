import numpy as np 
from scipy.special import digamma
from scipy.special import gamma
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def set_seed(s=42):
    np.random.seed(s)

    
class HPF(object):
    def __init__(self, X, K=5):
        
        self.K = K     # latent dimension
        self.a = 0.3   # Preference prior: Gamma shape parameter of theta ~ Gamma(a, psi_u)
        self.a1 = 0.3  # Activity Gamma shape paramter of psi_u ~ Gamma(a1, a1/b1)
        self.b1 = 1    # Activity Gamma rate paramter of psi_u ~ Gamma(a1, a1/b1)
        self.c = 0.3   # Attribute prior: Gamma shape parameter of beta ~ Gamma(c, eta_i)
        self.c1 = 0.3  # Popularity Gamma shape paramter of eta_i ~ Gamma(c1, c1/d1)
        self.d1 = 1    # Popularity Gamma rate paramter of eta_i ~ Gamma(c1, c1/d1)
        
        self.X = X
        self.n_U, self.n_I = np.shape(X)
        self.init() # initiaize parameters for VI
    
    def init(self):
        # initialize the variational parameters with a small random offset.
        # shape and rate param of activity kappa <- xi_u
        self.kappa_shp = self.a1 + self.K * self.a
        self.kappa_rte = self.a1 / self.b1 + np.random.uniform(0, 1, self.n_U)
        
        # shape and rate param of popularity tau <- eta_i
        self.tau_shp = self.c1 + self.K * self.c
        self.tau_rte = self.c1 / self.d1 + np.random.uniform(0, 1, self.n_I)
        
        # shape and rate param of preference gamma <- theta
        self.gamma_shp = self.a + np.random.uniform(0, 1, (self.n_U, self.K))
        self.gamma_rte = np.repeat(self.a / self.b1, self.K) + np.random.uniform(0, 1, (self.n_U, self.K))
        
        # shape and rate param of attribute lambda <- beta
        self.lambda_shp = self.c + np.random.uniform(0, 1, (self.n_I, self.K))
        self.lambda_rte = np.repeat(self.c / self.d1, self.K) + np.random.uniform(0, 1, (self.n_I, self.K))
        
        # param of rating phi <- y_ui
        self.phi = np.zeros((self.n_U, self.n_I, self.K))
    
    def variational_inference(self, n_epochs=20, early_stop=5, verbose=True):
        
        losses = [] # only for testing the convergence
        LJ = [] 
        best_loss = np.inf
        stop_cnt = 0
        
        # CAVI
        for epoch in tqdm(range(n_epochs)):
            
            for u, i in zip(self.X.nonzero()[0], self.X.nonzero()[1]):
                self.phi[u, i] = [
                    np.exp(
                        digamma(self.gamma_shp[u, k]) - np.log(self.gamma_rte[u, k]) +
                        digamma(self.lambda_shp[i, k]) - np.log(self.lambda_rte[i, k])
                    )
                    for k in range(self.K)
                ]
                self.phi[u, i] = self.phi[u, i] / np.sum(self.phi[u, i]) # normalize
            
            for u in range(self.n_U):
                self.gamma_shp[u] = [
                    self.a + self.X[u] @ self.phi[u, :, k]
                    for k in range(self.K)
                ]
                self.gamma_rte[u] = [
                    self.kappa_shp / self.kappa_rte[u] + np.sum(self.lambda_shp[:, k] / self.lambda_rte[:, k])
                    for k in range(self.K)
                ]
                self.kappa_rte[u] = self.a1 / self.b1 + np.sum(self.gamma_shp[u] / self.gamma_rte[u])
                
            for i in range(self.n_I):
                self.lambda_shp[i] = [
                    self.c + self.X[:, i] @ self.phi[:, i, k]
                    for k in range(self.K)
                ]
                self.lambda_rte[i] = [
                    self.tau_shp / self.tau_rte[i] + np.sum(self.gamma_shp[:, k] / self.gamma_rte[:, k])
                    for k in range(self.K)
                ]
                self.tau_rte[i] = self.c1 / self.d1 + np.sum(self.lambda_shp[i] / self.lambda_rte[i])
            
            # get prediction
            theta = self.gamma_shp / self.gamma_rte
            beta = self.lambda_shp / self.lambda_rte
            pred = theta @ beta.T
            self.theta, self.beta, self.pred = theta, beta, pred
            
            # evaluate
            lj = self.log_joint(theta, beta)
            loss = np.mean((pred.flatten() - self.X.flatten())**2)
            LJ.append(lj)
            losses.append(loss)
            if verbose:
                print(f"Epoch{epoch}: log joint={lj:.2f}, loss={loss:.2f}")
            
            # early stop on valid
            if loss < best_loss:
                best_loss = loss
                stop_cnt = 0
            
            stop_cnt += 1
            if stop_cnt >= early_stop:
                break
            
        return LJ, losses    
                
        
    def log_joint(self, theta, beta):
        res = 0
        pred = theta @ beta.T
        for u, i in zip(self.X.nonzero()[0], self.X.nonzero()[1]):
            res += (
                self.X[u, i] * np.log(pred[u, i]) 
                - np.log(gamma(self.X[u, i]))
            )
        res -= theta.sum(axis=0) @ beta.sum(axis=0).T
        return res

    
def accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions
    Count if output[topk] in target[topk]
    """
    res = []
    for k in topk:
        arr_pred = output.argsort(axis=1)[:,::-1][:,:k]
        arr_tar = target.argsort(axis=1)[:,::-1][:,:k]
        r = np.array([
            1 if len(np.intersect1d(arr_pred[i], arr_tar[i])) > 0 else 0 
            for i in range(len(arr_pred))
        ])
        res.append(r.mean())

    return res