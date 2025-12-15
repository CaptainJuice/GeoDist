# Geodesic Forests
**Research Track Paper**  
KDD '20, August 23–27, 2020, Virtual Event, USA  


## Authors
- Meghana Madhyastha: Department of Computer Science, Johns Hopkins University  
- Gongkai Li: Department of Applied Mathematics and Statistics, Johns Hopkins University  
- Veronika Strnadová-Neeley: Department of Computer Science, Montana State University  
- James Browne: Department of Computer Science, Johns Hopkins University  
- Joshua T. Vogelstein: Department of Biomedical Engineering, Johns Hopkins University  
- Randal Burns: Department of Computer Science, Johns Hopkins University  
- Carey E. Priebe: Department of Applied Mathematics and Statistics, Johns Hopkins University  


## ABSTRACT
Together with the curse of dimensionality, nonlinear dependencies in large data sets persist as major challenges in data mining tasks. A reliable way to accurately preserve nonlinear structure is to compute geodesic distances between data points. Manifold learning methods, such as Isomap, aim to preserve geodesic distances in a Riemannian manifold. However, as manifold learning algorithms operate on the ambient dimensionality of the data, the essential step of geodesic distance computation is sensitive to high-dimensional noise. Therefore, a direct application of these algorithms to high-dimensional, noisy data often yields unsatisfactory results and does not accurately capture nonlinear structure.

We propose an unsupervised random forest approach called **geodesic forests (GF)** to geodesic distance estimation in linear and nonlinear manifolds with noise. GF operates on low-dimensional sparse linear combinations of features, rather than the full observed dimensionality. To choose the optimal split in a computationally efficient fashion, we developed **Fast-BIC**, a fast Bayesian Information Criterion statistic for Gaussian mixture models. We additionally propose **geodesic precision and geodesic recall** as novel evaluation metrics that quantify how well the geodesic distances of a latent manifold are preserved. Empirical results on simulated and real data demonstrate that GF is robust to high-dimensional noise, whereas other methods (e.g., Isomap, UMAP, FLANN) quickly deteriorate in such settings. Notably, GF is able to estimate geodesic distances better than other approaches on a real connectome dataset.


## CCS CONCEPTS
- Computing methodologies → Dimensionality reduction and manifold learning  
- Theory of computation → Random projections and metric embeddings  
- Mathematics of computing → Probabilistic algorithms  


## KEYWORDS
noisy data, random forest, unsupervised, manifold learning  


## ACM Reference Format
Meghana Madhyastha, Gongkai Li, Veronika Strnadová-Neeley, James Browne, Joshua T. Vogelstein, Randal Burns, and Carey E. Priebe. 2020. Geodesic Forests. In 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20), August 23–27, 2020, Virtual Event, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3394486.3403094  


# 1 INTRODUCTION
Nearest neighbor algorithms, which sort all points according to their distances to one another, are considered among the top 10 most important data mining algorithms of all time [51] and have strong theoretical guarantees for both classification and regression [43]. Decision trees (such as CART and C4.5) can also reasonably be thought of as algorithms for organizing data in a hierarchical fashion; these are among the top ten algorithms as well [51]. Moreover, decision trees underlie both random forests [7] and gradient boosted trees [22], which are the two leading algorithms for machine learning on tabular data today [9–11]. There is a rich literature on approximate nearest neighbor algorithms (see Aumüller et al. [2] for benchmark comparisons), which are used extensively in big data systems.

However, operating on the exact nearest neighbors (or trying to approximate them) is not always desirable. For example, consider a simple supervised learning setting: Given a data corpus \(\{(x_{n}, y_{n})\}_{n=1}^{N}\), learn a decision rule that predicts \(y\) for a given new data point \(x\) and achieves a small error with high probability. A canonical approach is kernel regression [37]. A kernel machine’s prediction is a weighted linear combination of predictions from the neighbors of \(x\), specifically:  
\[
\hat{y}=\frac{1}{N} \sum_{n=1}^{N} y_{n} \times \kappa(x, x_{n})
\]  
for some suitably chosen kernel \(\kappa\) (e.g., a radial basis function or k-nearest neighbors kernel). Such approaches enjoy strong theoretical guarantees [33].  

Now, further assume that the \(x\)'s are noisy measurements of some true but unobserved \(\bar{x}\)'s. This assumption—called “measurement error modeling” [24]—is much more realistic than assuming \(x\)'s are noise-free [25]. Under this assumption, an ideal approach (with smaller error for the same sample size) would use kernel regression on the noise-free measurements:  
\[
\hat{y}=\frac{1}{N} \sum_{n=1}^{N} y_{n} \times \kappa(\bar{x}, \bar{x}_{n})
\]  
Unfortunately, \(\bar{x}\)'s are unobserved, so this approach is unavailable. This framework highlights the need to learn which sample points are close on an underlying latent structure (e.g., a manifold) for subsequent inference. Importantly, even for supervised tasks, performance may improve by learning only the structure of features \(x\) (ignoring labels \(y\)), as we demonstrate in this paper.

Learning latent structure becomes even more critical when the dimensionality \(p\) exceeds the sample size \(N\). In this case, an intermediate representation is required to avoid: (1) numerical instability in matrix operations, and (2) the “curse of dimensionality” for statistical operations. This representation can be **explicit** (e.g., manifold learning) or **implicit** (e.g., kernel machines). For large \(N\), even approximation algorithms are needed to compute quantities whose exact solution requires \(O(N^2)\) or \(O(N^3)\) space/time.

### Geodesic Learning Background
Geodesic distance is the shortest path between two points in a Riemannian manifold, and **geodesic learning** is the process of estimating these distances between pairs of points in a dataset. Given this definition, it is intuitive to use manifold learning algorithms to first learn the data’s latent structure, then estimate distances in that latent space. In fact, many manifold learning algorithms (e.g., Isomap) first estimate pairwise geodesic distances [27]. However, this is not common practice; instead, manifold learning (e.g., random projection) is often used to estimate distances in the observed, noisy, high-dimensional ambient space.  

Several fields have addressed related challenges:
- **Space-partitioning trees** (e.g., k-d trees) [14, 46] use binary recursive splits with hyperplanes for efficient geometric queries, but optimize for noisy observed data rather than latent noise-free structure.  
- **Decision trees** (and extensions like random forests [7] or gradient boosting [23]) are standard for supervised tasks but rarely used for unsupervised learning.  
- **Kernel learning analogies** with decision trees: Breiman [6] showed random forests are equivalent to a kernel on the true margin, a connection explored in recent literature [3, 17, 38, 39].  
- **Spectral manifold learning** (e.g., [36]) estimates pairwise geodesic distances first but operates on high-dimensional data and degrades with noise.  

None of these works explicitly define or estimate geodesic distances.


### Contributions of This Work
We propose **Geodesic Forests (GF)**, an unsupervised random forest approach that achieves near-linear space/time complexity while approximating true latent geodesic distances. Key innovations include:
1. **Sparse linear subspaces**: GF recursively clusters data in sparse linear subspaces (building on the “randomer forest” framework [48]) to separate signal from noise dimensions.  
2. **Fast-BIC**: A novel, efficient splitting criterion that computes the Bayesian Information Criterion (BIC) for 1D Gaussian mixture models (GMMs) exactly.  
3. **Geodesic precision/recall**: Novel evaluation metrics to quantify how well latent manifold geodesic distances are preserved (avoiding limitations of qualitative embedding visualization).  

Empirically, GF outperforms baselines (Isomap, UMAP, FLANN) on noisy data and a real Drosophila connectome dataset.


# 2 RELATED WORK
We review key methods for manifold learning, dimensionality reduction, and unsupervised forests—highlighting their limitations that GF addresses.

## 2.1 Manifold Learning (Geodesic Preservation)
- **Isomap [45]**: A classic method that preserves geodesic distances via three steps: (1) build a k-nearest neighbor/$\epsilon$-neighborhood graph (edges weighted by Euclidean distances), (2) compute all-pairs shortest paths, (3) embed into low dimensions. Limitations:  
  - Step 1 relies on high-dimensional Euclidean distances, which poorly estimate manifold distances.  
  - Space/time complexity: \(O(n^2)\) for pairwise distances, \(O(n^3)\) for shortest paths (prohibitive for large \(n\)).  
- **Laplacian Eigenmaps [4] / Hessian Eigenmaps [19] / Diffusion Maps [12]**: These address Isomap’s limitations by preserving alternative properties (e.g., diffusion distance) but still use Euclidean distances for neighbor graphs—failing in noisy high dimensions.  


## 2.2 Dimensionality Reduction (Embedding)
- **t-SNE [29, 30]**: Minimizes Kullback-Leibler divergence between high/low-dimensional neighbor distance distributions. Used primarily for visualization; cannot handle non-metric distances and scales poorly.  
- **UMAP [31]**: Extends t-SNE with a fuzzy simplicial set representation and force-directed embedding. Scales better than t-SNE but still relies on high-dimensional Euclidean distances for neighbor graphs—degrading with noise.  


## 2.3 Approximate Nearest Neighbors (ANN)
- **FLANN [34]**: Uses binary space-partitioning trees (e.g., k-d trees) to approximate high-dimensional nearest neighbors. Designed for observed space distances, not latent manifolds—fails with noise dimensions.  


## 2.4 Unsupervised Random Forests
- **Manifold Forests [13]**: Optimizes information gain via multivariate Gaussian differential entropy for splits. Computationally expensive; not evaluated on noisy data.  
- **Cutler’s RandomForest R Package [40]**: Generates synthetic data (feature-wise permutation) and classifies “real vs. synthetic” to learn structure. Misses simple latent patterns (as shown in our experiments).  


## 2.5 Random Projection Trees
- **Dasgupta & Freund [15, 16]**: Use random projection trees for manifold learning and vector quantization. Limitations: (1) random splits (no optimization), (2) single trees (no ensemble robustness). Their theoretical analysis motivates our geodesic precision metric.


# 3 GEODESIC FORESTS
GF extends the original random forest framework [7] with three key modifications (Section 3.0) and consists of four core components: algorithm overview (3.1), node-wise feature generation (3.2), splitting criteria (3.3), proximity matrix construction (3.4), and evaluation metrics (3.5).

## 3.0 Key Differences from Standard Random Forests
1. **Unsupervised learning**: GF learns latent structure without labels (unlike supervised classification/regression forests).  
2. **Fast-BIC splitting**: Replaces Gini impurity/entropy with a 1D GMM BIC criterion (exact, efficient).  
3. **Sparse oblique splits**: Uses random sparse linear combinations of features (not axis-aligned splits) to capture latent structure.  
4. **Proximity matrix fix**: Corrects a bug in standard implementations [28] where aggregated proximity matrices are not stochastic equivalents—improving efficiency and correctness.


## 3.1 GF Algorithm Overview
Given input data \(X = \{x_1, ..., x_N\}\) (each \(x_n \in \mathbb{R}^p\)), GF builds \(T\) decision trees via these steps:
1. **Tree initialization**: For each tree, sample a bootstrapped subset of size \(m < N\) from \(X\).  
2. **Recursive splitting**: For each node:  
   a. Generate \(d\) sparse linear feature combinations (Section 3.2).  
   b. Evaluate each feature via a splitting criterion (Section 3.3) to find the optimal split.  
   c. Split the node into left/right children; repeat until termination (e.g., minimum node size).  
3. **Proximity matrix**: After building all trees, compute a similarity matrix where \(S(i,j)\) is the fraction of trees in which points \(i\) and \(j\) share a leaf node (Section 3.4).  

Pseudocode for tree construction is provided in Appendix B (Algorithm 1).


## 3.2 Node-Wise Feature Generation
GF avoids axis-aligned splits (used in standard random forests) and instead uses **sparse oblique splits** (random linear combinations of features) to capture latent structure. For a node with data \(X' \subseteq X\):
1. **Sample projection matrix**: Generate a \(p \times d\) matrix \(A\) where entries are sampled from \(\{-1, +1\}\) (λpd times, then uniformly distributed) [48]. Here:  
   - \(λ = 1/20\) (controls sparsity of \(A\), following [48]).  
   - \(d\): Dimensionality of the projected space (tuned via validation).  
2. **Project data**: Transform \(X'\) into a \(d\)-dimensional space via \(\tilde{X} = A^T X'\). Each row \(\tilde{X}[i,:]\) is a 1D sparse linear combination of original features.  
3. **Evaluate splits**: For each row \(\tilde{X}[i,:]\), find the optimal split point (Section 3.3).


## 3.3 Splitting Criteria
GF evaluates three criteria to select the best split for each projected 1D feature. We focus on **Fast-BIC** (our novel criterion) as it balances speed and accuracy.

### 3.3.1 1. Two-Means Splitting
Minimizes the sum of intra-cluster variance for a 1D 2-means partition:  
\[
\min_{s} \sum_{n=1}^{s}\left(x_{n}-\hat{\mu}_{1}\right)^{2}+\sum_{n=s+1}^{N}\left(x_{n}-\hat{\mu}_{2}\right)^{2}
\]  
- **Steps**: Sort data, test splits between all sequential points, estimate cluster means via MLE.  
- **Limitation**: Ignores feature variance—zero-variance features always “win,” requiring ad-hoc rescaling (problematic for unsupervised learning).


### 3.3.2 2. Mclust-BIC Splitting
Fits a 2-component GMM to each projected feature and selects the split maximizing BIC.  
- **Steps**:  
  1. Use EM algorithm [21] to estimate GMM parameters (\(\mu_j, \sigma_j^2, \pi_j\)) and latent cluster memberships \(z_{n,j}\) (probability of point \(n\) in cluster \(j\)).  
  2. Compute BIC for the GMM:  
     \[
     BIC(M) = -2\ln(\hat{L}_M) + \ln(N) d_M
     \]  
     where \(\hat{L}_M\) = max log-likelihood of model \(M\), \(d_M\) = number of parameters (e.g., 5 for 2-component GMM: \(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \pi_1\)).  
  3. Split at the midpoint where the two Gaussians are equally likely.  
- **Limitations**: EM finds only local maxima (sensitive to initialization) and converges slowly [32].


### 3.3.3 3. Fast-BIC Splitting (Novel)
Combines the speed of two-means with the model flexibility of Mclust-BIC. Explicitly computes BIC for 1D 2-component GMMs via hard clustering (avoiding EM).  

#### Key Ideas:
- **Hard clustering**: For each sorted split \(s\), assign points left of \(s\) to cluster 1, right to cluster 2 (no soft memberships).  
- **Exact MLE estimation**: For each split, compute cluster parameters via MLE:  
  \[
  \hat{\mu}_1 = \frac{1}{s}\sum_{n \leq s}x_n, \quad \hat{\sigma}_1^2 = \frac{1}{s}\sum_{n \leq s}\|x_n - \hat{\mu}_1\|^2, \quad \hat{\pi}_1 = \frac{s}{N}
  \]  
  (equivalent for cluster 2, using \(N-s\) points).  
- **BIC computation**: Evaluate BIC for two cases (same/different variances) and choose the minimum:  
  1. **Different variances**: BIC uses \(\sigma_1^2 \neq \sigma_2^2\).  
  2. **Same variance**: BIC uses a pooled variance \(\hat{\sigma}_{\text{comb}}^2 = \frac{1}{N}\sum_{j=1}^2\sum_{n \in C_j}\|x_n - \hat{\mu}_j\|^2\).  

#### Advantage Over Mclust-BIC:
- **Exact global maximum**: No EM—tests all splits, guaranteeing the optimal BIC.  
- **Speed**: Runs as fast as two-means (avoids EM iterations).  

Pseudocode for Fast-BIC is provided in Appendix B (Algorithm 3).


## 3.4 Proximity Matrix Construction
The proximity matrix \(S \in \mathbb{R}^{N \times N}\) quantifies similarity between points via their co-occurrence in tree leaves:  
\[
S(i,j) = \frac{L_{i,j}}{T_{i,j}}
\]  
where:
- \(L_{i,j}\): Number of trees where points \(i\) and \(j\) share a leaf node.  
- \(T_{i,j}\): Number of trees where both \(i\) and \(j\) are in the bootstrapped sample.  

We use both in-bag and out-of-bag samples to ensure all point pairs are evaluated. This matrix implicitly captures latent geodesic distances—higher \(S(i,j)\) indicates closer geodesic proximity.


## 3.5 Geodesic Precision and Recall (Novel Metrics)
Existing manifold learning evaluations rely on qualitative embeddings or downstream tasks (e.g., classification)—failing to directly measure geodesic preservation. We introduce **geodesic precision** and **geodesic recall** to compare estimated neighbors to true latent neighbors.

### Definitions
For a query point \(x\), corpus \(D_N = \{x_1, ..., x_N\}\), and query size \(k\):
- **Relevant neighbors**: The \(k\) points in \(D_N\) closest to \(x\) via **true latent geodesic distance** (unknown in practice; we use ground truth for simulations, cell types for connectomes).  
- **Retrieved neighbors**: The \(k\) points closest to \(x\) via the learner’s estimated distance (e.g., GF’s proximity matrix).  

Metrics are defined as:  
\[
\text{Geodesic Precision} = \frac{|\{\text{Relevant Neighbors}\} \cap \{\text{Retrieved Neighbors}\}|}{|\{\text{Retrieved Neighbors}\}|}
\]  
\[
\text{Geodesic Recall} = \frac{|\{\text{Relevant Neighbors}\} \cap \{\text{Retrieved Neighbors}\}|}{|\{\text{Relevant Neighbors}\}|}
\]  

### Key Properties
- Averaged over all query points to get dataset-level scores.  
- **Continuous manifolds**: Finite geodesic distances between all pairs (e.g., helix, sphere).  
- **Discrete manifolds**: Clusters with no connections (e.g., Gaussian mixture)—precision and recall are identical (neighbors = same cluster).  
- Tight bounds: Geodesic precision bounds downstream classification accuracy [18]—poor precision implies poor manifold inference.


# 4 NUMERICAL RESULTS
We evaluate GF on four simulated manifolds (Section 4.1) and a real connectome dataset (Section 4.4). Baselines include: Isomap, UMAP, FLANN, Euclidean distance, and unsupervised random forests (URF—axis-aligned splits).

## 4.1 Simulation Settings
We use four synthetic manifolds (Figure 1) to cover linear/nonlinear, connected/disconnected cases:
| Manifold   | Description                                                                 | Latent Dimension |
|------------|-----------------------------------------------------------------------------|------------------|
| Linear     | \(p = (4t, 6t, 9t)\), \(t \in (0,1)\) (equally spaced grid)                 | 1                |
| Helix      | \(p = (t\cos t, t\sin t, t)\), \(t \in (2\pi, 9\pi)\) (swiss roll analog)    | 1                |
| Sphere     | \(p = (9\cos u\sin v, 9\sin u\sin v, 9\cos v)\), \(u \in (0,2\pi), v \in (0,\pi)\) | 2 |
| Gaussian Mixture | 3 Gaussians: \(\mu_1 = [-3,-3,-3], \mu_2 = [0,0,0], \mu_3 = [3,3,3]\), \(\Sigma = I\) | Discrete (3 clusters) |

All simulations use \(N=1000\) points, 3 signal dimensions, and add Gaussian noise (\(y_n \sim \mathcal{N}(0, 70I)\)) in \(d'\) dimensions (varied from 2 to 10,000).


## 4.2 Splitting Criterion Selection & Hyperparameter Robustness
### 4.2.1 Splitting Criterion Comparison (Figure 2)
We compare three criteria (two-means, Mclust-BIC, Fast-BIC) for GF (sparse oblique splits) and URF (axis-aligned splits):
- **BIC-based criteria** (Fast-BIC, Mclust-BIC) outperform two-means in all cases.  
- **GF outperforms URF**: Sparse oblique splits better capture latent structure than axis-aligned splits.  
- **Fast-BIC = Mclust-BIC accuracy, faster speed**: We use Fast-BIC for all后续 experiments.

### 4.2.2 Hyperparameter Robustness (Figure 3)
GF’s performance is stable to changes in key hyperparameters:
- **minparent**: Minimum node size (5–45). Geodesic precision remains >0.6 for all values.  
- **mtry**: Number of projected features tested per node (\(\sqrt{d}\), \(d/2\), \(d\)). No significant performance drop.  

We fix **minparent=100** and **mtry=\(\sqrt{d}\)** for后续 experiments.


## 4.3 GF is Robust to Noise Dimensions (Figure 4)
We measure geodesic precision@k=50 as noise dimensions (\(d'\)) increase from 2 to 10,000:
- **GF**: Maintains high precision (>0.6) across all noise levels (even 10,000 noise dimensions).  
- **Baselines**: Degrade to chance (precision ~0.05) as \(d'\) increases:  
  - Fastest degradation: FLANN, Euclidean distance (rely on high-dimensional distances).  
  - Moderate degradation: Isomap, UMAP (neighbor graphs use Euclidean distances).  

Even after normalizing features (0–1 scaling, bottom panel of Figure 4), GF outperforms baselines—proving its noise robustness is not due to scale sensitivity.


## 4.4 GF on Drosophila Connectome (Figure 5)
We test GF on the **larval Drosophila mushroom body connectome** [20]—a brain network with:
- 200 nodes (neurons), ~75,000 edges (synapses).  
- 4 cell types: Kenyon Cells (KC), Input Neurons (MBIN), Output Neurons (MBON), Projection Neurons (PN).  
- Latent representation: 6-dimensional embedding via adjacency spectral embedding [35] (3 “outgoing,” 3 “incoming” dimensions).  

### Evaluation:
Treat cell type as true latent neighbors (relevant neighbors = same cell type). GF achieves **higher recall at all precision levels** (Figure 5, bottom):
- GF recall: ~0.9 at precision=0.8.  
- Baselines (Isomap, UMAP, FLANN): Recall <0.7 at the same precision.  

This demonstrates GF’s utility for real-world high-dimensional biological data.


# 5 DISCUSSION
## 5.1 Key Takeaways
1. **Noise robustness**: GF’s sparse oblique splits and Fast-BIC criterion enable it to separate signal from noise—outperforming baselines on high-dimensional noisy data.  
2. **Unsupervised flexibility**: GF works on linear/nonlinear, continuous/discrete manifolds without labels.  
3. **Direct evaluation**: Geodesic precision/recall avoid limitations of qualitative embeddings and downstream task reliance.  


## 5.2 Future Directions
- **Theoretical analysis**: Extend random projection tree guarantees [15, 16] to GF; derive bounds on geodesic learning for Bayes optimal performance.  
- **Supervised geodesic learning**: Adapt GF to use labels for task-specific geodesic estimation (e.g., recommendation systems).  
- **Broader applications**: GF can improve manifold learning, clustering, anomaly detection, and vertex nomination [52]—all of which rely on accurate distance estimation.  


## 5.3 Software Availability
GF is open-source as part of the **SPORF** package (Python/R): https://neurodata.io/sporf/


# ACKNOWLEDGMENTS
This work was supported by DARPA’s D3M program and Lifelong Learning Machines program (contract FA8650-18-2-7834).


# REFERENCES
[1] David Arthur and Sergei Vassilvitskii. 2007. k-means++: The advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. 1027–1035. https://dl.acm.org/citation.cfm?id=128338  
[2] Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2017. ANNBenchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms. In Similarity Search and Applications. Springer, 34–49. https://doi.org/10.1007/978-3-319-68474-1_3  
[3] Matej Balog, Balaji Lakshminarayanan, Zoubin Ghahramani, Daniel M Roy, and Yee Whye Teh. 2016. The Mondrian Kernel. arXiv:stat.ML/1606.05241. http://arxiv.org/abs/1606.05241  
[4] Mikhail Belkin and Partha Niyogi. 2002. Laplacian eigenmaps and spectral techniques for embedding and clustering. In Advances in neural information processing systems. 585–591.  
[5] Gérard Biau, Luc Devroye, and Gábor Lugosi. 2008. Consistency of random forests and other averaging classifiers. Journal of Machine Learning Research 9, Sep (2008), 2015–2033.  
[6] Leo Breiman. 2000. Some infinity theory for predictor ensembles. Technical Report 579, Statistics Dept. UCB. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.24.7078&rep=rep1&type=pdf  
[7] Leo Breiman. 2001. Random forests. Machine learning 45, 1 (2001), 5–32.  
[8] Leo Breiman, Jerome Friedman, Charles J Stone, and R A Olshen. 1984. Classification and Regression Trees. Chapman and Hall/CRC.  
[9] Rich Caruana, Nikos Karampatziakis, and Ainur Yessenalina. 2008. An empirical evaluation of supervised learning in high dimensions. In Proceedings of the 25th international conference on Machine learning. ACM, 96–103. https://doi.org/10.1145/1390156.1390169  
[10] Rich Caruana and Alexandru Niculescu-Mizil. 2006. An Empirical Comparison of Supervised Learning Algorithms. In ICML ’06. ACM, 161–168. https://doi.org/10.1145/1143844.1143865  
[11] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In KDD ’16. ACM, 785–794. https://doi.org/10.1145/2939672.2939785  
[12] Ronald R Coifman and S Lafon. 2006. Diffusion maps. Appl. Comput. Harmon. Anal. 21, 1 (2006), 5–30.  
[13] A Criminisi and J Shotton. 2013. Manifold forests. In Decision Forests for Computer Vision and Medical Image Analysis. Springer, 79–93.  
[14] Sanjoy Dasgupta and Yoav Freund. 2008. Random projection trees and low dimensional manifolds. In STOC ’08. Citeseer, 537–546.  
[15] Sanjoy Dasgupta and Yoav Freund. 2008. Random projection trees and low dimensional manifolds. In STOC ’08. Citeseer, 537–546.  
[16] Sanjoy Dasgupta and Yoav Freund. 2008. Random projection trees for vector quantization. IEEE Trans. Inf. Theory 54, 5 (2008), 3229–3242. arXiv:stat.ML/0805.1390  
[17] Alex Davies and Zoubin Ghahramani. 2014. The Random Forest Kernel and other kernels for big data from random partitions. arXiv:stat.ML/1402.4293. http://arxiv.org/abs/1402.4293  
[18] Luc Devroye, Laszlo Györfi, and Gabor Lugosi. 1997. A Probabilistic Theory of Pattern Recognition. Springer.  
[19] David L Donoho and Carrie Grimes. 2003. Hessian eigenmaps: locally linear embedding techniques for high-dimensional data. Proc. Natl. Acad. Sci. U. S. A. 100, 10 (2003), 5591–5596.  
[20] Katharina Eichler et al. 2017. The complete connectome of a learning and memory centre in an insect brain. Nature 548, 7666 (2017), 175.  
[21] Chris Fraley and Adrian E Raftery. 2002. Model-Based Clustering, Discriminant Analysis, and Density Estimation. J. Amer. Statist. Assoc. 97, 458 (2002), 611–631. https://doi.org/10.1198/016214502760047131  
[22] Y Freund and R E Schapire. 1997. A decision-theoretic generalization of online learning and an application to boosting. J. Comput. System Sci. 55, 1 (1997), 119–139.  
[23] Jerome H Friedman. 2001. Greedy function approximation: a gradient boosting machine. Annals of statistics (2001), 1189–1232.  
[24] Wayne A Fuller. 1987. Measurement Error Models. Wiley.  
[25] David J Hand. 2016. Measurement: A Very Short Introduction. Oxford University Press.  
[26] J A Hartigan and M A Wong. 1979. Algorithm AS 136: A K-Means Clustering Algorithm. Applied statistics 28, 1 (1979), 100. https://doi.org/10.2307/2346830  
[27] John A Lee and Michel Verleysen. 2007. Nonlinear Dimensionality Reduction. Springer Science & Business Media.  
[28] Andy Liaw and Matthew Wiener. 2002. Classification and Regression by randomForest. R News 2, 3 (2002), 18–22.  
[29] Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing Data using t-SNE. J. Mach. Learn. Res. 9, Nov (2008), 2579–2605.  
[30] Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of machine learning research 9, Nov (2008), 2579–2605.  
[31] Leland McInnes and John Healy. 2018. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.  
[32] Geoffrey McLachlan and Thriyambakam Krishnan. 2008. The EM Algorithm and Extensions. Wiley-Interscience.  
[33] Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. 2018. Foundations of Machine Learning. MIT Press.  
[34] Marius Muja and David G Lowe. 2014. Scalable nearest neighbor algorithms for high dimensional data. IEEE Transactions on Pattern Analysis & Machine Intelligence 36, 11 (2014), 2227–2240.  
[35] Carey E Priebe et al. 2017. Semiparametric spectral modeling of the Drosophila connectome. arXiv preprint arXiv:1705.03297.  
[36] Bernhard Schölkopf, Alexander Smola, and Klaus-Robert Müller. 1997. Kernel principal component analysis. In International Conference on Artificial Neural Networks. Springer, 583–588.  
[37] Bernhard Schölkopf and Alexander J Smola. 2002. Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press.  
[38] E Scornet. 2016. Random Forests and Kernel Methods. IEEE Trans. Inf. Theory 62, 3 (2016), 1485–1500. https://doi.org/10.1109/TIT.2016.2514489  
[39] Cencheng Shen and Joshua T Vogelstein. 2018. Decision Forests Induce Characteristic Kernels. arXiv:stat.ML/1812.00029. http://arxiv.org/abs/1812.00029  
[40] Tao Shi and Steve Horvath. 2006. Unsupervised Learning With Random Forest Predictors. J. Comput. Graph. Stat. 15, 1 (2006), 118–138.  
[41] Tao Shi and Steve Horvath. 2006. Unsupervised learning with random forest predictors. Journal of Computational and Graphical Statistics 15, 1 (2006), 118–138.  
[42] Vin D Silva and Joshua B Tenenbaum. 2003. Global Versus Local Methods in Nonlinear Dimensionality Reduction. In Advances in Neural Information Processing Systems 15. MIT Press, 721–728.  
[43] Charles J Stone. 1977. Consistent Nonparametric Regression. Annals of statistics 5, 4 (1977), 595–620. https://doi.org/10.1214/aos/1176343886  
[44] Daniel L Sussman et al. 2012. A consistent adjacency spectral embedding for stochastic blockmodel graphs. J. Amer. Statist. Assoc. 107, 499 (2012), 1119–1128.  
[45] Joshua B Tenenbaum, Vin De Silva, and John C Langford. 2000. A global geometric framework for nonlinear dimensionality reduction. Science 290, 5500 (2000), 2319–2323.  
[46] William C Thibault and Bruce F Naylor. 1987. Set operations on polyhedra using binary space partitioning trees. In ACM SIGGRAPH computer graphics, Vol. 21. ACM, 153–162.  
[47] T Tomita, M Maggioni, and J Vogelstein. 2017. ROFLMAO: Robust Oblique Forests with Linear MAtrix Operations. In SIAM International Conference on Data Mining. 498–506. https://doi.org/10.1137/1.9781611974973.56  
[48] Tyler M Tomita, Mauro Maggioni, and Joshua T Vogelstein. 2015. Randomer forests. arXiv preprint arXiv:1506.03410.  
[49] Joshua T Vogelstein et al. 2019. Connectal coding: discovering the structures linking cognitive phenotypes to individual histories. Current opinion in neurobiology 55 (2019), 199–212.  
[50] Joe H Ward. 1963. Hierarchical Grouping to Optimize an Objective Function. J. Amer. Statist. Assoc. 58, 301 (1963), 236–244. https://doi.org/10.1080/01621459.1963.10500845  
[51] Xindong Wu et al. 2007. Top 10 Algorithms in Data Mining. Knowledge and information systems 14, 1 (2007), 1–37. https://doi.org/10.1007/s10115-007-0114-2  
[52] Jordan Yoder et al. 2018. Vertex nomination: The canonical sampling and the extended spectral nomination schemes. arXiv preprint arXiv:1802.04960.  


# APPENDICES
## A SUPPLEMENTAL FIGURES
Figure 6 (not shown here) extends Section 4.3 to low noise dimensions (\(d'=2\)–10, unnormalized data). GF maintains high geodesic precision (>0.7) across all manifolds, while baselines degrade to <0.3—confirming GF’s noise robustness.


## B ALGORITHMS
### Algorithm 1: Build a Geodesic Decision Tree
```plaintext
procedure BuildTree(X, d, Θ)
    Input:
        X: Subset of training data (dimension p)
        d: Dimensionality of projected space
        Θ: Split eligibility criteria (e.g., min node size)
    Output: Tree t

    if Θ not satisfied then
        return LeafNode(X)  // Create leaf node
    else
        // Sample random sparse projection matrix
        A ← p×d matrix, entries sampled from {-1,+1} (λ=1/20 sparsity)
        // Project data to d-dimensional space
        X̃ = Aᵀ X
        min_t* ← ∞
        bestDim ← 1
        splitPoint ← 0

        // Evaluate each projected dimension for optimal split
        for i ∈ {1, ..., d} do
            X̃(i) ← X̃[:, i]  // Extract i-th projected feature
            (midpt, t*) = ChooseSplit(X̃(i))  // Use Algorithm 2 (two-means) or 3 (Fast-BIC)
            if t* < min_t* then
                min_t* ← t*
                bestDim ← i
                splitPoint ← midpt
            end if
        end for

        // Split data into left/right children
        Xleft = {x ∈ X | x(bestDim) < splitPoint}
        Xright = {x ∈ X | x(bestDim) ≥ splitPoint}

        // Recursively build child trees
        Daughters.Left = BuildTree(Xleft, d, Θ)
        Daughters.Right = BuildTree(Xright, d, Θ)

        return Daughters
    end if
end procedure
```

### Algorithm 3: Fast-BIC1D (Optimal Split for 1D Data)
```plaintext
procedure Fast-BIC1D(Z)
    Input: Z ∈ ℝⁿ (1D data points)
    Output: (splitPoint, minBIC) (best partition midpoint and BIC score)

    // Initialize clusters
    μ̂₁ ← min(Z)
    C₁ ← {μ̂₁}
    C₂ ← Z \ C₁
    μ̂₂ ← (1/|C₂|) ∑_{z ∈ C₂} z  // Mean of C₂
    minBIC ← ∞

    while C₂ ≠ ∅ do
        // Move smallest point from C₂ to C₁
        z ← min(C₂)
        C₁ ← C₁ ∪ {z}
        C₂ ← C₂ \ {z}

        // Estimate cluster parameters (MLE)
        for j = 1, 2 do
            nⱼ ← |Cⱼ|
            ŵⱼ ← nⱼ / n  // Prior probability
            μ̂ⱼ ← (1/nⱼ) ∑_{z ∈ Cⱼ} z  // Mean
            σ̂ⱼ² ← (1/nⱼ) ∑_{z ∈ Cⱼ} ||z - μ̂ⱼ||²  // Variance
        end for

        // Pooled variance (same-variance case)
        σ̂_comb² ← (1/n) [∑_{z ∈ C₁} ||z - μ̂₁||² + ∑_{z ∈ C₂} ||z - μ̂₂||²]

        // Compute BIC for both variance cases
        BIC_diff_var ← -2[ n₁logŵ₁ - (n₁/2)log(2πσ̂₁²) - (n₁/2) + 
                           n₂logŵ₂ - (n₂/2)log(2πσ̂₂²) - (n₂/2) ] + 
                       log(n) * 5  // 5 parameters: μ₁,μ₂,σ₁²,σ₂²,ŵ₁
        BIC_same_var ← -2[ n₁logŵ₁ - (n₁/2)log(2πσ̂_comb²) - (n₁/2) + 
                           n₂logŵ₂ - (n₂/2)log(2πσ̂_comb²) - (n₂/2) ] + 
                       log(n) * 4  // 4 parameters: μ₁,μ₂,σ̂_comb²,ŵ₁

        // Select minimum BIC
        BIC_curr ← min(BIC_diff_var, BIC_same_var)

        // Update best split if current BIC is lower
        if BIC_curr < minBIC then
            minBIC ← BIC_curr
            splitPoint ← (max(C₁) + min(C₂))/2  // Midpoint between clusters
        end if
    end while

    return (splitPoint, minBIC)
end procedure
```


## C SIMULATION SETTINGS (Detailed)
- **Linear**: \(t\) sampled from 1000 equally spaced values in (0,1); \(p = (4t, 6t, 9t)\).  
- **Helix**: \(t\) sampled from 1000 equally spaced values in (2π, 9π); \(p = (t\cos t, t\sin t, t)\).  
- **Sphere**: \(u\) (azimuth) sampled from 1000 equally spaced values in (0,2π); \(v\) (polar angle) sampled from 1000 equally spaced values in (0,π); \(p = (9\cos u\sin v, 9\sin u\sin v, 9\cos v)\).  
- **Gaussian Mixture**: 1000 points total, 333/333/334 from each Gaussian; \(\mu_1 = [-3,-3,-3], \mu_2 = [0,0,0], \mu_3 = [3,3,3]\), \(\Sigma = I\) (identity matrix).


## D FAST-BIC1D DERIVATION
Fast-BIC computes the log-likelihood of a 2-component GMM with hard clustering (no EM):  
\[
\log \ell(Z) = \sum_{z \in C₁} [\logŵ₁ + \log\mathcal{N}(z; \mû₁, σ̂₁²)] + \sum_{z \in C₂} [\logŵ₂ + \log\mathcal{N}(z; \mû₂, σ̂₂²)]
\]  
Substituting MLE parameters (\(\hat{w}_j = n_j/n\), \(\hat{\mu}_j = \sum z/n_j\), \(\hat{\sigma}_j^2 = \sum ||z-\hat{\mu}_j||^2/n_j\)) and simplifying, the negative log-likelihood becomes:  
\[
-2\log\ell = n₁log(2πσ̂₁²) + n₂log(2πσ̂₂²) - 2n₁logŵ₁ - 2n₂logŵ₂ + n
\]  
BIC adds a regularization term (\(\log(n) \times d_M\), where \(d_M\) = number of parameters) to penalize model complexity. For the same-variance case, \(d_M=4\); for different variances, \(d_M=5\).
