---
title: The genealogy of language models 
date: 2026-08-06
description: An account of language model progress. 
---

# The genealogy of language models
This is my foray into surveying the landscape of modern language models. The intention is to recover all modern techniques and ideas and consolidate them here. The motivation behind doing so is rather selfish, in that I want to learn about this and understand why—this is not actively meant as an end-all resource. Also as I imagine as I delve further in this process, I will veer to topics that are adjacent but not at the core of language models; moreso, techniques employed for SOTA models to make them better.

## Attention is all you need
:::cite
Vaswani et al. **Attention Is All You Need.** 2017. [arXiv](https://arxiv.org/abs/1706.03762)
:::

Let us begin with the seminal work: the reason I no longer write emails and every individual contributor has become, in effect, a manager of small artificial employees.

For posterity, I will note that the paper did not invent "Attention", nor was the Transformer introduced as the decoder-only language model we now colloquially mean when we say "Transformer." Attention had already become common in sequence-to-sequence models, particularly for machine translation, where it was generally attached to a recurrent network. All I will attribute to this paper is dispensing recurrence and convolution in models, instead agruing for constructing models out of attention (mostly).

To understand why this mattered, consider the then-prevalent [recurrent nueral newtowrk](https://en.wikipedia.org/wiki/Recurrent_neural_network). An RNN reads a sequence one token at a time and repeatedly updates a hidden state: 
$$
	h_t=f(h_{t-1}, x_t).
$$
Here, $x_t$ is the representation of the token at position $t$ and $h_t$ is the model’s summary of everything it has read up until position $1,\ldots,t$.

This is an incredibly intuitive way to think about how we think about language: we read the first word, update our understanding, read the second word, and continue. Unfortunately, it is irreducibly sequential. To compute $h_{100}$, the model must first compute $h_{99}$; to compute $h_{99}$, it must first compute $h_{98}$. Even during training, when the entire sentence is already known, the hidden states form a dependency chain that cannot be evaluated all at once.

Another problem that the neural net must learn to deal with is communicating information about distant tokens. Consider the sentence
>"Because the trophy was too large for the suitcase, it did not fit."
When we read the word *it*, both *trophy* and *suitacse* are gramatically plausible antecedents. We know from the phrase "too large for" is what makese it so that *trophy* is the more referent. A recurrent model can represent this relationship, but information concerning trophy must survive every intervening update before it can influence the representation of it. [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) and [GRUs](https://en.wikipedia.org/wiki/Gated_recurrent_unit) were designed in part to make these long-range dependencies easier to preserve, but they did not remove the sequential path itself.

The Transformer replaces this chain with direct communication. Instead of requiring information at one position to be repeatedly propagated to every position between it and its destination, self-attention permits one position to directly retrieve information from another.

### Attention
The central operation is easier to understand as retrieval than as cognition. Suppose a token needs information from the rest of the sequence. It emits a **query** describing what it is looking for. Every token exposes a **key** describing what kind of information it contains and a **value** containing the information it will return if selected.

For each token representation $x_i \in \mathbb{R}^{d_{\text{model}}}$, the model learns three linear projections:
$$
	q_i = x_i W^Q, \quad k_i = x_i W^K, \quad v_i = x_i W^V.
$$
The words **query, key,** and **value** should not be taken too literally and are not indicative of any linguistic concepts (as far as I know). The matrices $W^Q$, $W^K$, and $W^V$ are learned through gradient descent. If resolving pronouns is useful for predicting text, the attention mechanism may learn queries and keys that make pronouns compatible with plausible antecedents.

The compatibility between a query $q_i$ and a key $k_j$ is measured by their dot product:
$$
	s_{ij} = q_i k_j^\top.
$$
The scores are converted into a probability distribution with a softmax:
$$
	\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{m=1}^{n} \exp(s_{im})}.
$$
Finally, token $i$ receives a weighted average of the values: 
$$
	z_i = \sum_{j=1}^{n} \alpha_{ij}v_j.
$$
This weighted average $z_i$ is the output of attention for token $i$.

#### A worked example
Let’s go back to our sentence
> "The chicken did not cross the street because it had no legs."

Suppose, the query emitted by "it" is
$$
	q_{\text{it}} = \begin{bmatrix}1 & 0\end{bmatrix},
$$
and the keys being considered are
$$
	K = 
	\begin{bmatrix}
	1.0 & 0.0 \\
	0.1 & 0.2 \\
	0.0 & 1.0 \\
	0.8 & 0.1
	\end{bmatrix},
$$
corresponding to *animals, cross, street,* and *legs*. The unscaled dot products are
$$
	q_{\text{it}}K^\top =
	\begin{bmatrix}
		1.0 & 0.1 & 0.0 & 0.8
	\end{bmatrix}.
$$
We can then apply softmax and obtain a distribution where most of the mass is found in *chicken* and *legs*. Attention constructs a new representation of **it** by retrieving a mixture of information from positions that its query finds relevant.

```python
import numpy as np

query = np.array([[1.0, 0.0]])
keys = np.array(
    [
        [1.0, 0.0],  # animal
        [0.1, 0.2],  # cross
        [0.0, 1.0],  # street
        [0.8, 0.1],  # tired
    ]
)

scores = query @ keys.T
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)

print(weights.round(3))
```

#### Scaling terms
The paper's actual attention equation includes a scaling term:
$$
	\operatorname{Attention}(Q,K,V)= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$
Suppose the coordinates of $q$ and $k$ are independent, centered random variables i.e. mean is zero with variance one. Their dot product is
$$
	q \cdot k = \sum_{r=1}^{d_k} q_r k_r.
$$
Thus, each product has variance approximately one and the variance of the sum is roughly $d_k$. Its typical magnitude would consequently grow $\sqrt{d_k}$. As the key dimension increases, unscaled dot products become larger in magnitude.

Large logits push softmax toward nearly one-hot distributions where exactly one element is one and all other elements are zero, representing categorical data or discrete outcomes as vectors, making optimizaition fragile. Dividing by $\sqrt{d_k}$ keeps the score scale roughly stable.
:::progress
