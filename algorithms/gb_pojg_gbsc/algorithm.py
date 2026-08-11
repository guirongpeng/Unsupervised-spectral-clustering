from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from core.algorithm import Algorithm as BenchmarkAlgorithm
from algorithms.gb_pojg_gbdpc.generator import GBPOJGGenerationResult, GranularBall, generate_granular_balls
from .config import GBPOJGGBSCConfig

@dataclass(frozen=True)
class GBPOJGGBSCResult:
    labels: np.ndarray; similarity: np.ndarray; eigenvalues: np.ndarray; embedding: np.ndarray; ball_labels: np.ndarray; generation: GBPOJGGenerationResult

def run_gb_pojg_gbsc(X: np.ndarray, n_clusters: int, config: GBPOJGGBSCConfig, seed: int = 1) -> GBPOJGGBSCResult:
    values=np.asarray(X,dtype=float)
    if values.ndim!=2 or values.shape[0]<2 or values.shape[1]<1 or not np.all(np.isfinite(values)): raise ValueError("X must be a finite nonempty 2-D feature matrix")
    generation=generate_granular_balls(values,config.gamma,config.delta); balls=generation.granular_balls
    if n_clusters<2 or n_clusters>len(balls): raise ValueError("n_clusters exceeds granular-ball count")
    centers=np.asarray([ball.center for ball in balls]); radii=np.asarray([ball.average_radius for ball in balls]); distance=np.maximum(cdist(centers,centers)-radii[:,None]-radii[None,:],0)
    similarity=np.exp(-(distance**2)/(2*config.sigma**2)); degree=similarity.sum(axis=1)
    laplacian=np.eye(len(balls))-similarity/np.sqrt(degree[:,None]*degree[None,:])
    eigenvalues,eigenvectors=eigh(laplacian,subset_by_index=(0,n_clusters-1),check_finite=False); norms=np.linalg.norm(eigenvectors,axis=1)
    if np.any(norms==0): raise RuntimeError("zero spectral embedding row")
    embedding=eigenvectors/norms[:,None]; ball_labels=KMeans(n_clusters=n_clusters,n_init=config.n_init,max_iter=config.max_iter,random_state=seed).fit_predict(embedding)+1
    labels=np.zeros(values.shape[0],dtype=int)
    for ball,label in zip(balls,ball_labels): labels[ball.sample_indices]=label
    return GBPOJGGBSCResult(labels,similarity,eigenvalues,embedding,ball_labels,generation)

class GBPOJGGBSC(BenchmarkAlgorithm):
    def __init__(self,config:GBPOJGGBSCConfig,n_clusters:int,random_state:int=1): self.config=config; self.n_clusters=n_clusters; self.random_state=random_state
    def fit(self,X:np.ndarray)->"GBPOJGGBSC":
        self.result_=run_gb_pojg_gbsc(X,self.n_clusters,self.config,self.random_state); self.labels_=self.result_.labels; self.granular_balls_=self.result_.generation.granular_balls; return self
    def get_params(self)->dict[str,object]: return {"n_clusters":self.n_clusters,"random_state":self.random_state,**asdict(self.config)}
