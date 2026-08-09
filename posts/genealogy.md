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

For posterity, I will note that the paper did not invent "Attention", nor was the Transformer introduced as the decoder-only language model we now colloquially mean when we say "Transformer." Attention had already become common in sequence-to-sequence models, particularly for machine translation, where it was generally attached to a recurrent network. All I will attribute to this paper is dispensing recurrence and convolution in models, instead arguing for constructing models out of attention (mostly).

### Motivation
To understand why this mattered, consider the then-prevalent [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network). An RNN reads a sequence one token at a time and repeatedly updates a hidden state: 
$$
	h_t=f(h_{t-1}, x_t).
$$
Here, $x_t$ is the representation of the token at position $t$ and $h_t$ is the model’s summary of everything it has read up until position $1,\ldots,t$.

This is an incredibly intuitive way to think about how we think about language: we read the first word, update our understanding, read the second word, and continue. Unfortunately, it is irreducibly sequential. To compute $h_{100}$, the model must first compute $h_{99}$; to compute $h_{99}$, it must first compute $h_{98}$. Even during training, when the entire sentence is already known, the hidden states form a dependency chain that cannot be evaluated all at once.

Another problem that the neural net must learn to deal with is communicating information about distant tokens. Consider the sentence
>"Because the trophy was too large for the suitcase, it did not fit."
When we read the word *it*, both *trophy* and *suitcase* are grammatically plausible antecedents. We know from the phrase "too large for" is what makes it so that *trophy* is the more referent. A recurrent model can represent this relationship, but information concerning trophy must survive every intervening update before it can influence the representation of it. [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) and [GRUs](https://en.wikipedia.org/wiki/Gated_recurrent_unit) were designed in part to make these long-range dependencies easier to preserve, but they did not remove the sequential path itself.

The Transformer replaces this chain with direct communication. Instead of requiring information at one position to be repeatedly propagated to every position between it and its destination, self-attention permits one position to directly retrieve information from another.

### Attention
The central operation is easier to understand as retrieval than as cognition. Suppose a token needs information from the rest of the sequence. It emits a **query** describing what it is looking for. Every token produces a **key**, against which queries can be compared, and a **value**, containing the information that will actually be retrieved.

For each token representation $x_i \in \mathbb{R}^{d_{\text{model}}}$, the model learns three linear projections:
$$
	q_i = x_i W^Q, \quad k_i = x_i W^K, \quad v_i = x_i W^V.
$$
The words **query, key,** and **value** should not be taken too literally and are not indicative of any linguistic concepts (as far as I know). That is, there is no column in $W^Q$ that corresponds to say pronoun antecedent. These matrices begin as parameters and are learned through gradient descent. If some rule or subtask is useful to the task, the attention mechanism may learn queries and keys that make pronouns compatible with plausible antecedents.

Given a query $q_i$ and the key $k_j$ of some other token, compatibility is measured as
$$
	s_{ij} = q_i k_j^\top.
$$
A larger dot product means that, in the representation learned by this particular attention head, the two vectors are more compatible. The scores are converted into a probability distribution with a softmax:
$$
	\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{m=1}^{n} \exp(s_{im})}.
$$
The resulting $\alpha_{ij}$ are nonnegative and sum to one. Finally, we use them to form a weighted average of the values:
$$
	z_i = \sum_{j=1}^{n} \alpha_{ij}v_j.
$$
That weighted sum is the output of attention for position $i$.

So attention does two separate things:
1. $QK^\top$ determines where to retrieve from.
2. Multiplication by $V$ determines what information is retrieved.

#### A worked example
Let’s go back to our sentence
> "Because the trophy was too large for the suitcase, it did not fit."
Imagine that we are looking at one attention head in some later layer of the network. By this point, the representation of *it* contains contextual information from previous layers. This head has learned something useful for resolving which earlier object the current token is referring to.

Suppose, the query emitted by "it" is
$$
	q_{\text{it}} = \begin{bmatrix}2 & 1 & -1\end{bmatrix}.
$$
And suppose four earlier positions expose the following keys:
$$
	K = 
	\begin{bmatrix}
		1.8 & 0.8 & -0.8\\
		0.2 & 0.1 & 0.5\\
		0.3 & 0.5 & 1.3\\
		0 & 0 & 0
	\end{bmatrix}
$$
corresponding respectively to *trophy, large, suitcase,* and *because*. I have deliberately chosen tiny vectors so that we can work out the arithmetic; the individual coordinates should not be interpreted as actual concepts learned by a Transformer.

The dot product is
$$
	q_{\text{it}}K^\top = \begin{bmatrix} 5.2 & 0 & -0.2 & 0 \end{bmatrix}
$$
Already, the query is much more compatible with trophy than with suitcase. After applying the scaling factor ($\sqrt{d_k}$) used in the actual Transformer,
$$
	\frac{q_{\text{it}}K^\top}{\sqrt{3}} \approx \begin{bmatrix} 3 & 0 & -0.12 & 0 \end{bmatrix}
$$
Softmax turns these scores into approximately
$$
	\begin{bmatrix} 0.874 & 0.043 & 0.039 & 0.043 \end{bmatrix}
$$
Informally, we can conclude that $87\%$ of this head's retrieval is coming from the representation associated with *trophy*. We produce this computation in python in the following snippet
```python
import numpy as np

query = np.array([[2.0, 1.0, -1.0]])
keys = np.array(
    [
        [1.8, 0.8, -0.8],  # trophy
        [0.2, 0.1, 0.5],  # large
        [0.3, 0.5, 1.3],  # suitcase
        [0.0, 0.0, 0.0],  # because
    ]
)

scores = (query @ keys.T) / np.sqrt(keys.shape[-1])
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
```
Of course, attending strongly to *trophy* is useless unless there is something useful to retrieve from it. This is the role of the values. If its corresponding value is $v_{\text{trophy}}$, then it receives approximately
$$
	 0.874v_{\text{trophy}} + 0.043v_{\text{large}} + 0.039v_{\text{suitcase}} + 0.043v_{\text{because}}.
$$
The representation at *it* has therefore been updated with information largely taken from *trophy*.

#### Scaled dot-product attention
The paper's actual attention equation is
$$
	\operatorname{Attention}(Q,K,V)= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$
Rather than computing attention for one query at a time, we stack every query, key, and value into a matrix
$$
	Q \in \mathbb{R}^{n \times d_k}, K \in \mathbb{R}^{n \times d_k}, V \in \mathbb{R}^{n \times d_v},
$$
respectively. Then, $QK^\top \in \mathbb{R}^{n \times n}$ is populated by the compatibility score between every pair of positions. Row $i$ tells us where position $i$ wants to look. After applying the row-wise softmax, multiplying by $V$ performs all of the operations simultaneously.

The last part of the question is the scaling factor $\sqrt{d_k}$. Suppose the coordinates of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Then
$$
	qk = \sum_{i=1}^{d_k}q_ik_i,
$$
where each product has variance $1$; thus, the variance of the sum is $\operatorname{Var}(qk) = d_k$ and consequently the standard deviation is $\sqrt{d_k}$. As $d_k$ grows, the dot products naturally become larger in magnitude. Given enough large logits, softmax and its output become  one-hot distributions where exactly one element is one and all other elements are zero, representing categorical data or discrete outcomes as vectors, making optimization fragile. The paper introduces the $1/\sqrt{d_k}$ factor precisely to counteract this effect.
:::progress 2026-08-08T12:18:45.429Z

### Multi-head attention
In the example worked above, the attention mechanism offers a response to a single question, "what does *it* refer to?" In any given sentence, it seems almost undeniable that there are a plethora of relations between each token, and it would be in our best interest to model most of them. Compressing all of those patterns into a single weighted average would force them to compete.

Therefore, the Transformer therefore runs several attention operations in parallel. For head $i$:
$$
	\operatorname{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V).
$$
The outputs are concatenated and projected back into the model dimension:
$$
	\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\operatorname{head}_1, \ldots, \operatorname{head}_h)W^O,
$$
:::note
The base Transformer used $d_{\text{model}} = 512, h=8, d_k=d_v=64$. Notice that $h \cdot d_k = d_{\text{model}}$
:::
where each head gets its own learned projections and therefore its own representation space in which to decide what counts as relevant. The representation is projected into smaller spaces for each head, the heads operate in parallel, and their outputs are recombined.

### Self-attention
The term self-attention simply means that $Q$, $K$, and $V$ originate from the same sequence.

Let $X = \langle x_1, \ldots, x_n \rangle$. Then we know $Q=XW^Q,K=XW^K,V=XW^V$. Every position therefore asks a question of every position in the same sequence including itself.

This is subtly different from the attention mechanisms that had commonly appeared in encoder-decoder systems before the Transformer. There, attention often allowed the decoder to look back at representations produced by the encoder. The Transformer retains this kind of cross-attention, but also uses attention as the mechanism by which representations inside the encoder and decoder themselves communicate.

The original architecture consequently contains three forms of attention:
1. encoder self-attention
2. masked decoder self-attention
3. encoder-decoder cross-attention.
That distinction matters because modern language models will eventually discard two-thirds of this architecture (can you guess which one?).

#### A conundrum
There is an immediate problem with replacing recurrence: positionality. An RNN gets position almost for free. The first token is processed first, the second token second, and so forth. Its computation itself contains an ordering; self-attention does not.

If we ignore the positions of our tokens, then permuting the rows of $X$ simply permutes the rows of the output, but that shouldn't be the case. We want the sentence
> dog bites man
to mean a completely different thing than the sentence
> man bites dog.
Thus, we need to add positions into the process. The paper does this by adding a positional encoding to each token embedding before it enters the Transformer stack.

### Positional encoding
For position $\operatorname{pos}$ and dimension $i$, define the following sinusoidal functions
$$
	\operatorname{PE}_{(\operatorname{pos}, 2i)} = \sin \left( \frac{\operatorname{pos}}{10000^{2i/d_{\text{model}}}} \right)
$$
and
$$
	\operatorname{PE}_{(\operatorname{pos}, 2i+1)} = \cos \left( \frac{\operatorname{pos}}{10000^{2i/d_{\text{model}}}} \right).
$$
By design, the positional vector produced has the same dimensionality as the token embedding, so we can just set $x_{\operatorname{pos}} = e_{\text{token}} + \operatorname{PE}_{\operatorname{pos}}$.

Recall your trig identities
$$
	\sin((\alpha + \beta)\omega) = \sin(\alpha\omega)\cos(\beta\omega) + \cos(\alpha\omega)\cos(\beta\omega)
$$
and
$$
	\cos((\alpha + \beta)\omega) = \cos(\alpha\omega)\cos(\beta\omega) - \sin(\alpha\omega)\sin(\beta\omega).
$$
Thus, once the sine and cosine at position $p$ are known, the encoding at position $p+k$ can be expressed as a linear transformation of the functions whose coefficients depend only on $k$: "for any fixed offset $k$, $\operatorname{PE}_{p+k}$ can be represented as a linear function of $\operatorname{PE}_p$, potentially making relative positions easy for attention layers to learn."

As we move forward, positional encoding will be reinvented repeatedly.

### The rest
The original Transformer is an encoder-decoder model designed primarily for sequence transduction. For a given input sequence of symbol representations $x = \langle x_1, \ldots, x_n \rangle$ it produces a sequence of continuous representation $z = \langle z_1, \ldots, z_n \rangle$. The decoder then autoregressively generates an output sequence $y = \langle y_1, \ldots, y_m \rangle$.

Each encoder layer contains:
1. multi-head self-attention
2. a position-wise feed-forward network
Each decoder layer contains:
1. masked multi-head self-attention
2. multi-head cross-attention over the encoder
3. a position-wise feed-forward network

Around every sub-layer, the original Transformer uses a residual connection followed by layer normalization: $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$.
:::progress 2026-08-08T13:13:29.654Z
