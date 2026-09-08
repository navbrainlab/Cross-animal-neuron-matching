# NeuRID architecture and objective

## 1. Population-relative multimodal identity formation

For animal \(s\in\{A,B\}\), coordinates are centered and scaled within the
animal. Activity traces are standardized within each neuron. Geometry node
features \(g_i^s\) and activity node features \(a_i^s\) are fused with a learned
convex mixture and a zero-initialized interaction residual.

The geometry relation edge contains the relative displacement, distance, and
multi-scale radial fields:

\[
e^{G,s}_{ik}=
[\Delta x_{ik};d_{ik};
\exp(-d_{ik}^2/2\sigma_1^2),\ldots,
\exp(-d_{ik}^2/2\sigma_K^2)].
\]

The activity relation contains within-animal trace and derivative
correlations. Learned geometry and activity edges are fused symmetrically into
\(e^s_{ik}\). No identity labels or cross-animal candidates are used to create
these relations.

A relation-biased self-attention encoder is applied separately:

\[
h_i^s=\mathcal E_\theta
\left(f_i^s,\{f_k^s,e^s_{ik}\}_{k=1}^{N_s}\right).
\]

The final low-dimensional relation field is

\[
U^s_{ik}=\tanh W_R[e^s_{ik};h_i^s-h_k^s;h_i^s\odot h_k^s].
\]

This is the explicit representation of a neuron's multimodal role relative to
its own population.

## 2. Cross-population relational transport

Only after both encodings have been formed independently is the unary score
computed:

\[
S_{ij}=\frac{\langle \bar h_i^A,\bar h_j^B\rangle}{\tau_u}.
\]

An augmented Sinkhorn plan \(P^{(0)}\) initializes correspondence. At refinement
step \(t\), relational inconsistency is

\[
D_{ij}(P^{(t)})=\sum_{k,l}\bar P^{(t)}_{kl}
\|U^A_{ik}-U^B_{jl}\|_2^2,
\]

and the next plan is obtained from

\[
L^{(t+1)}=S-\lambda D(P^{(t)})/\tau_r.
\]

The temperatures are positive by construction and the structural weight is
learned through a softplus parameterization.

The contraction is evaluated exactly as

\[
\begin{aligned}
D_{ij}={}&\sum_k r_k\|U^A_{ik}\|^2
+\sum_l c_l\|U^B_{jl}\|^2\\
&-2\sum_q U^{A,q}P(U^{B,q})^\top,
\end{aligned}
\]

where \(r=P\mathbf 1\) and \(c=P^\top\mathbf 1\). This removes the otherwise
prohibitive four-index tensor.

## 3. Capacity-correct dustbin

For \(N_A\) and \(N_B\) real nodes, the augmented row and column marginals are

\[
\tilde\mu=\frac{[1,\ldots,1,N_B]}{N_A+N_B},\qquad
\tilde\nu=\frac{[1,\ldots,1,N_A]}{N_A+N_B}.
\]

The non-unit dustbin capacities are necessary: uniform marginals would allow a
single dustbin to absorb only one node. Deletion and insertion logits are
learned separately. The complete augmented plan is retained for training.

## 4. Single final supervision objective

For real row \(i\), convert transport mass to a categorical distribution using
\(p_i=\tilde P_{i,:}/\tilde\mu_i\); the analogous column distribution is
\(q_j=\tilde P_{:,j}/\tilde\nu_j\). The symmetric focal objective is

\[
\mathcal L=-\frac{1}{|\mathcal Q|}
\sum_{q\in\mathcal Q}(1-p_{q,y_q})^\gamma
\log(p_{q,y_q}+\epsilon).
\]

`gamma=0` is ordinary categorical cross-entropy. The loss supervises the final
structured assignment only. Modality-specific representations and relations
are learned because they change the final plan, not through extra label losses.

Original labels absent from the other animal are masked because an unlabeled
counterpart may exist. Dustbin targets are supervised only after the code
deliberately removes one member of a known homologous pair.
