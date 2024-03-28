$\mathcal{L}_{LSCE}=(1-\epsilon)*[-\sum_{k=1}^k\log(\mathcal{p_k})\mathcal{q_k}]+\epsilon*[-\sum_{k=1}^k\log(\mathcal{p_k})\frac{1}{\mathcal{k}}]=(1-\epsilon)*\mathcal{H(q,p)}+\epsilon*\mathcal{H(u,p)} , \mathcal{u(k)}=\frac{1}{\mathcal{k}}$

$\mathcal{L}_{FL}=-\alpha_t(1-\mathcal{p_t})\log(\mathcal{p_t})$



$\mathcal{L}_{CE}=-\sum_{k=1}^k\log(\mathcal{p_k})\mathcal{q_k}$

$\mathcal{L}_{LSCE}=(1-\epsilon)*[-\sum_{k=1}^k\log(\mathcal{p_k})\mathcal{q_k}]+\epsilon*[-\sum_{k=1}^k\log(\mathcal{p_k})\frac{1}{\mathcal{k}}]$

$\mathcal{L}_{FL}=-\sum_{k=1}^k\alpha(1-\mathcal{p_k})^\gamma\log(\mathcal{p_k})\mathcal{q_k}$

$\mathcal{L}_{LSFL}=(1-\epsilon)*[-\sum_{k=1}^k\alpha(1-\mathcal{p_k})^\gamma\log(\mathcal{p_k})\mathcal{q_k}]+\epsilon*[-\sum_{k=1}^k\log(\mathcal{p_k})\frac{1}{\mathcal{k}}]$
$$
\mathcal{p_t}=
\begin{cases}
\widehat{p}\,\,\,\,\,\,\,\,\,\,\,\,,\,\,y=1\\
1-\widehat{p}\,\,,\,\,otherwise\\
\end{cases}
$$
