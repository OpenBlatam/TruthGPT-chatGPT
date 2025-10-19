New architectures:

https://arxiv.org/html/2507.20096v2

### 2 Distance-Based Attention Score  
In the Transformer model, the scaled dot-product attention mechanism is defined as follows. Given an input sequence embedding \(X\), which includes positional encodings, we compute the query, key, and value matrices as:  
\[
Q = X W^{(Q)}, \quad K = X W^{(K)}, \quad V = X W^{(V)},
\]  
where \(W^{(Q)}, W^{(K)}, W^{(V)}\) are learnable weight matrices. The attention score matrix \(\alpha \in \mathbb R^{N\times N}\) is then calculated using:  
\[
\alpha = \mathrm{softmax}\!\Big(\frac{Q\,K^\top}{\sqrt{d_k}}\Big),
\]  
and the output of the attention layer is:  
\[
O = \alpha\,V.
\]  
Each output vector \(O_i\) is a weighted sum of the value vectors:  
\[
O_i = \sum_{j=1}^N \alpha_{ij}\;V_j,
\]  
where \(\alpha_{ij} = \frac{\exp\!\big(Q_i \cdot K_j / \sqrt{d_k}\big)}{\sum_{j'} \exp\!\big(Q_i \cdot K_{j'} / \sqrt{d_k}\big)}\).  

The dot-product \(Q_i \cdot K_j\) can be reformulated in terms of the \(L_2\) distance:  
\[
\langle Q_i, K_j \rangle = \tfrac12 \big(\|Q_i\|_2^2 + \|K_j\|_2^2 - \|Q_i - K_j\|_2^2\big).
\]  
If queries and keys are properly normalized with unit \(L_2\) norms, then  
\[
\alpha_{ij} = \frac{\exp\!\big(-\tfrac{1}{2\,\sqrt{d_k}}\;\|Q_i - K_j\|_2^2\big)}{\sum_{j'} \exp\!\big(-\tfrac{1}{2\,\sqrt{d_k}}\;\|Q_i - K_{j'}\|_2^2\big)}.
\]  
This equation demonstrates that the attention weight is functionally dependent on the distances between the queries and keys.  

Thus, we propose a distance-based attention mechanism constructed as follows. Given \(Q\) and \(K\) matrices, we construct an operator \(L\) such that \(L(Q,K)\) generates a matrix where  
\[
L_{ij} = -\,\mathrm{distance}(Q_i,\,K_j).
\]  
The new attention \(\alpha_{\text{new}} \in \mathbb R^{N\times N}\) is computed by  
\[
\alpha_{\text{new}} = \mathrm{softmax}\!\Big(\frac{\lambda\,L}{\sqrt{d_k}}\Big), \tag{1}
\]  
and  
\[
O = \alpha_{\text{new}}\,V, \tag{2}
\]  
where \(\lambda\) is a tuning parameter. Many distance measures can be applied, including the general \(L_p\) distance with \(p \ge 1\). In this paper, we focus on the attention mechanism based on the \(L_1\) distance, where  
\[
L_{ij} = -\,\|Q_i - K_j\|_1 = - \sum_{m=1}^{d_k} \big|Q_{i,m} - K_{j,m}\big|.
\]  


