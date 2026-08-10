---
title: The genealogy of language models 
date: 2026-08-06
description: An account of language model progress. 
---

# The genealogy of language models
This is my foray into surveying the landscape of modern language models. My aim is to recover the techniques and ideas behind modern systems and consolidate them here. The motivation is rather selfish: I want to understand why these systems work. This is not intended to be an end-all resource. As I delve further into the subject, I expect to veer into adjacent topics—especially the techniques used to improve state-of-the-art models.

## Attention is all you need
Let us begin with the seminal work: :::cite Vaswani et al. **Attention Is All You Need.** 2017. [arXiv](https://arxiv.org/abs/1706.03762) ::: It is also the reason I no longer write emails and every individual contributor has become, in effect, a manager of small artificial employees.

For posterity, I will note that the paper did not invent *attention*, nor was the Transformer introduced as the decoder-only language model we now colloquially mean when we say "Transformer." Attention had already become common in sequence-to-sequence models, particularly for machine translation, where it was generally attached to a recurrent network. The paper's novelty was to ask whether recurrence and convolution were necessary at all. Its answer was, mostly, no—and at the current moment, that answer seems to have held up.

The original Transformer was an encoder-decoder architecture for sequence transduction. That is, it ingested a sentence in one language, constructed a representation of it, and autoregressively :::definition An **autoregressive** model generates a sequence one item at a time, conditioning each new item on those that came before it. ::: produced the corresponding sentence in another. It became the foundation for many modern language models not only because it performed well across a variety of tasks, but also because it replaced the sequential core of previous architectures with operations that can be parallelized.

### Motivation
To understand why this mattered, consider the then-prevalent [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network). An RNN reads a sequence one token at a time and repeatedly updates a hidden state:
$$
	h_t=f(h_{t-1}, x_t).
$$
Here, $x_t$ is the representation of the token at position $t$, while $h_t$ is the model's summary of what it has read through position $t$.

This is an intuitive model of how we read language: we read the first word, update our understanding, read the second word, and continue. Unfortunately, it is irreducibly sequential. To compute $h_{100}$, the model must first compute $h_{99}$; to compute $h_{99}$, it must first compute $h_{98}$. Even during training, when the entire sentence is already known, the hidden states form a dependency chain that cannot be evaluated all at once.

GPUs are extraordinary at parallel workloads. An RNN instead presents them with a sequence of dependent operations, one that involves waiting for a previous calculation.

There is a second problem. Consider the sentence
> "Because the trophy was too large for the suitcase, it did not fit."
When we read the word *it*, both *trophy* and *suitcase* are grammatically plausible antecedents. The phrase *too large for* is what makes *trophy* the sensible referent. A recurrent model can represent this relationship, but information concerning *trophy* must survive every intervening update before it can influence the representation of *it*. [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) and [GRUs](https://en.wikipedia.org/wiki/Gated_recurrent_unit) were designed in part to make these long-range dependencies easier to preserve, but they did not remove the sequential path itself. In a recurrent network, information at position $i$ may need to pass through $O(|j-i|)$ intermediate states before affecting position $j$.

The Transformer proposes something much more direct:
> let every position communicate with every other position.

### Attention
The central operation is easier to understand as retrieval than as cognition. Suppose a token needs information from the rest of the sequence. It emits a **query** describing what it is looking for. Every token exposes a **key**, against which queries can be compared, and a **value**, containing the information that will actually be retrieved.

For each token representation $x_i \in \mathbb{R}^{d_{\text{model}}}$, the model learns three linear projections:
$$
	q_i = x_i W^Q, \quad k_i = x_i W^K, \quad v_i = x_i W^V.
$$
The words **query, key,** and **value** should not be taken too literally. They are not predefined linguistic objects. There is no column of $W^Q$ labelled *pronoun antecedent* or one column of $W^K$ labelled *subject of sentence*. The matrices begin as parameters and are learned through gradient descent. If identifying antecedents is useful for minimizing the training objective, the model may discover representations in which certain queries become highly compatible with certain keys.

Given the query $q_i$ at one position and the key $k_j$ at another, a compatibility score is measured as
$$
	s_{ij} = q_i k_j^\top.
$$
A large value means, loosely, that position $j$ appears relevant to whatever position $i$ is looking for.

We convert the collection of scores into nonnegative weights using softmax:
$$
	\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{m=1}^{n} \exp(s_{im})}.
$$
The resulting weights satisfy $\alpha_{ij} \ge 0$ and $\sum_j \alpha_{ij}=1$.


Finally, we retrieve the corresponding values:
$$
	z_i = \sum_{j=1}^{n} \alpha_{ij}v_j.
$$
That weighted sum is the output of attention for position $i$.

So attention does two conceptually distinct things:
1. $QK^\top$ determines **where information should come from**.
2. Multiplication by $V$ determines **what information is actually passed onward**.

#### A worked example
Let’s go back to our sentence
> "Because the trophy was too large for the suitcase, it did not fit."
Imagine that we are looking at one attention head in some later layer of the network. By this point, the representation of *it* contains contextual information from previous layers. This head has learned something useful for resolving which earlier object the current token is referring to.

Suppose the query emitted by "it" is
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
corresponding respectively to *trophy*, *large*, *suitcase*, and *because*. :::note  I have deliberately chosen tiny vectors so that we can work out the arithmetic. :::
The individual coordinates should not be interpreted as actual concepts learned by a Transformer.

The compatibility scores are
$$
	q_{\text{it}}K^\top = \begin{bmatrix} 5.2 & 0 & -0.2 & 0 \end{bmatrix}
$$
Already, the query is much more compatible with *trophy* than with *suitcase*. The actual Transformer scales these scores before applying softmax:
$$
	\frac{q_{\text{it}}K^\top}{\sqrt{3}} \approx \begin{bmatrix} 3 & 0 & -0.12 & 0 \end{bmatrix}
$$
Softmax turns these scaled scores into approximately
$$
	\begin{bmatrix} 0.874 & 0.043 & 0.039 & 0.043 \end{bmatrix}
$$
Informally, about $87\%$ of this head's retrieval is coming from the representation associated with *trophy*. We can reproduce this computation in Python:
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
Of course, attending strongly to *trophy* is useless unless there is something useful to retrieve from it. This is the role of the values. If its corresponding value is $v_{\text{trophy}},v_{\text{large}},\ldots$, the resulting vector is
$$
	 0.874v_{\text{trophy}} + 0.043v_{\text{large}} + 0.039v_{\text{suitcase}} + 0.043v_{\text{because}}.
$$
The representation at *it* has therefore been updated with information largely from *trophy*.

Attention does not copy another token's representation. It retrieves a learned projection of that representation, $v_j=x_jW^V$, and mixes it with projected information from other positions. What gets communicated is itself learned.

#### Scaled dot-product attention
The paper's actual attention equation is
$$
	\operatorname{Attention}(Q,K,V)= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$
Rather than computing one query at a time, we pack all queries, keys, and values into matrices. For a self-attention layer over a sequence of length $n$,
$$
	Q \in \mathbb{R}^{n \times d_k}, K \in \mathbb{R}^{n \times d_k}, V \in \mathbb{R}^{n \times d_v}.
$$
Then, $QK^\top \in \mathbb{R}^{n \times n}$. Entry $(i,j)$ is the compatibility between the query at position $i$ and the key at position $j$. Row $i$ therefore contains every position from which position $i$ might retrieve information.

Attention mechanisms before the Transformer often used a small neural network to compute compatibility between a query and a key. Dot-product attention instead reduces the entire problem to matrix multiplication.

Softmax then converts each row into positive weights summing to one. We finally multiply by $V$ to obtain the corresponding weighted sums of value vectors—the output of the attention mechanism.
> The keys are not themselves the information being returned. Queries and keys determine where information flows; values determine what flows.

The last part of the equation is the scaling factor $1/\sqrt{d_k}$.

Suppose the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Their dot product is
$$
	q \cdot k = \sum_{i=1}^{d_k}q_ik_i,
$$
where each product has variance $1$; thus, the variance of the sum is $\operatorname{Var}(q \cdot k) = d_k$ and consequently the standard deviation is $\sqrt{d_k}$.

As dimensionality $d_k$ grows, the dot products naturally become larger in magnitude. Applying softmax to large logits can push the resulting distribution toward a nearly one-hot vector, with most of the probability mass concentrated on one element. In those saturated regions, gradients through the softmax can become extremely small, making optimization difficult. Dividing by $\sqrt{d_k}$ keeps the scale of the logits roughly stable as $d_k$ grows and counteracts this effect.

### Multi-head attention
> One set of attention weights produces one weighted average.

In the example worked above, the attention mechanism offers a response to a single question, "what does *it* refer to?" But language contains many relationships simultaneously. A token may require information about position, syntax, punctuation, grammar, and any number of other features. Compressing all of those patterns into a single weighted average would force them to compete.

The Transformer therefore runs several attention operations in parallel. For head $i$:
$$
	\operatorname{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V).
$$
The outputs are concatenated and projected back into the model dimension:
$$
	\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\operatorname{head}_1, \ldots, \operatorname{head}_h)W^O.
$$
Each head receives its own learned projections $W_i^Q,W_i^K,W_i^V$. It therefore gets its own representation space in which to decide what counts as relevant and what information ought to be returned. :::note The base Transformer used $d_{\text{model}} = 512, h=8, d_k=d_v=64$. Notice that $h \cdot d_k = d_{\text{model}}$. Thus each head works in a $64$-dimensional query/key/value space, and the eight $64$-dimensional outputs concatenate back to $512$ dimensions before the final output projection. :::

This construction gives us a better way of understanding an attention head. A head is not merely "looking at a token." It consists of:
1. a learned query projection defining what a position searches for
2. a learned key projection defining how candidate positions advertise themselves
3. a learned value projection defining what information those positions provide
4. an attention pattern deciding where to retrieve from
Different heads can therefore implement different communication patterns over the same sequence.

There is a temptation here to anthropomorphize individual heads. Some heads do display remarkably interpretable patterns, and the original paper includes examples of heads tracking syntactic relationships. But nothing in the architecture requires one head to correspond cleanly to one linguistic or human concept. The safer interpretation is simply that
> multi-head attention gives the model multiple learned communication channels.

### Self-attention
The term **self-attention** describes where $Q$, $K$, and $V$ come from.

Let
$$
	X = \langle x_1, \ldots, x_n \rangle \in \mathbb{R}^{n \times d_{\text{model}}}.
$$
Suppressing the head index, in self-attention,
$$
	Q = XW^Q, K = XW^K, V = XW^V.
$$
The queries, keys, and values are all constructed from representations belonging to the same sequence. In encoder self-attention, every position may attend to every other position, including itself.

This is subtly different from the attention mechanisms that had commonly appeared in encoder-decoder systems before the Transformer. There, attention often allowed the decoder to look back at representations produced by the encoder. The Transformer retains this operation, which we will call **cross-attention**, but also uses attention as the mechanism through which positions inside the encoder and decoder communicate with one another.

The original architecture consequently contains three applications of multi-head attention:
1. encoder self-attention
2. masked decoder self-attention
3. encoder-decoder cross-attention
That distinction will matter enormously in a moment because the architecture eventually inherited by GPT discards the encoder and, with it, cross-attention entirely.

But first we have created a problem.

#### The positionality problem
There is an immediate problem with replacing recurrence: positionality. An RNN gets position almost for free. The first token is processed first, the second token second, and so forth. Its computation itself contains an ordering; self-attention does not.

Ignoring positional information, suppose we permute the sequence:
$$
	X' = PX
$$
where $P$ is some permutation matrix. :::note A permutation matrix is a square binary matrix that has exactly one entry of 1 in each row and each column, with all other entries being 0. Multiplying another matrix by a permutation matrix reorders its rows or columns. ::: Then,
$$
	Q' = X'W^Q = PQ, \quad K' = X'W^K = PK, \quad V' = X'W^V = PV.
$$
The attention scores become
$$
	Q'K'^\top = PQK^\top P^\top. 
$$
Because row-wise softmax respects the same permutation, multiplying the resulting weights by $V'=PV$ permutes the output in exactly the corresponding way. In other words, self-attention by itself knows that certain token representations exist but not where they occur in the sequence.

This poses a problem because
> dog bites man
does not mean the same thing as
> man bites dog.
Thus, we need to add positions deliberately into the process.

### Positional encoding
The paper does this by adding a positional vector to each token embedding before it enters the Transformer stack.

For position $\operatorname{pos}$ and dimension $i$,
$$
	\operatorname{PE}_{(\operatorname{pos}, 2i)} = \sin \left( \frac{\operatorname{pos}}{10000^{2i/d_{\text{model}}}} \right)
$$
and
$$
	\operatorname{PE}_{(\operatorname{pos}, 2i+1)} = \cos \left( \frac{\operatorname{pos}}{10000^{2i/d_{\text{model}}}} \right).
$$
Because the positional encoding has dimension $d_{\text{model}}$, it can simply be added to the token embedding:
$$
	x_{\operatorname{pos}} = e_{\text{token}} + \operatorname{PE}_{\operatorname{pos}}.
$$

The first question one might reasonably ask is: why trigonometric functions?

Each adjacent pair of positional dimensions forms a sine-cosine pair at some angular frequency $\omega$:
$$
	\begin{bmatrix}
		\sin(\rho\omega)\\
		\cos(\rho\omega)
	\end{bmatrix}
$$
Recall your trig identities
$$
	\sin((\alpha + \beta)\omega) = \sin(\alpha\omega)\cos(\beta\omega) + \cos(\alpha\omega)\sin(\beta\omega)
$$
and
$$
	\cos((\alpha + \beta)\omega) = \cos(\alpha\omega)\cos(\beta\omega) - \sin(\alpha\omega)\sin(\beta\omega).
$$
We can therefore write
$$
	\begin{bmatrix}
		\sin((\alpha + \beta)\omega) \\
		\cos((\alpha + \beta)\omega)
	\end{bmatrix}
	=
	\begin{bmatrix}
		\cos(\beta\omega) & \sin(\beta\omega) \\
		-\sin(\beta\omega) & \cos(\beta\omega)
	\end{bmatrix}
	\begin{bmatrix}
		\sin(\alpha\omega)\\
		\cos(\alpha\omega)
	\end{bmatrix}
$$
For any fixed displacement $\beta$, moving from the representation of position $\alpha$ to the representation of position $\alpha+\beta$ is therefore a linear transformation whose coefficients depend only on the displacement.

The model does not merely receive a unique identifier for every position. The encoding is structured such that relationships like "five tokens earlier" can, in principle, be recovered using simple linear operations.

The different dimensions use different frequencies. Some oscillate relatively quickly and distinguish nearby positions; others change very slowly and preserve information over larger distances. Taken together, they give every position a structured, multi-scale representation.

As we move forward, positional encoding will be reinvented repeatedly.

### The rest of the Transformer
Technically, **attention is not, in fact, all you need.**

The Transformer also contains feed-forward networks, residual connections, normalization, embeddings, positional information, and an output projection.

The original Transformer is an encoder-decoder model. Given an input sequence of symbol representations
$$
	x = \langle x_1, \ldots, x_n \rangle
$$ 
the encoder constructs a sequence of continuous representations
$$
	z = \langle z_1, \ldots, z_n \rangle.
$$
Conditioned on these representations, the decoder autoregressively generates an output sequence
$$
	y = \langle y_1, \ldots, y_m \rangle.
$$
The base model contains six encoder layers and six decoder layers.

We will discuss what happens inside each.

#### The position-wise feed-forward network
Every encoder and decoder layer contains, in addition to attention, a small fully connected neural network:
$$
	\operatorname{FFN}(x) = \operatorname{ReLU}(xW_1 + b_1)W_2 + b_2.
$$
In the paper, the first projection expands the dimension fourfold: $d_{\text{model}} = 512$ and $d_{\text{ff}} = 2048$. The network then applies ReLU, replacing negative values with zero, before the second matrix multiplication projects the result back into the model dimension.

This network is applied independently to every sequence position, using the same parameters at every position. Given a matrix $X$, the FFN transforms each row separately. There is no communication between tokens inside the feed-forward sublayer.
> attention mixes information across positions; the feed-forward network transforms information within each position.

#### Residual connection
Stacking nonlinear transformations creates another problem. Every layer is now responsible for taking the representation from the previous layer and replacing it with something *better*. As networks get deeper, forcing every block to relearn the entire representation can make optimization difficult. The Transformer therefore surrounds each sublayer with a residual connection.

Instead of computing $\operatorname{SubLayer}(x)$, it computes $x + \operatorname{SubLayer}(x)$. The input now has a direct path around the transformation.

A sublayer need not produce an entirely new representation from scratch; it can learn a correction or update to what already exists. This perspective will become conceptually important when we eventually discuss mechanistic interpretability and the residual stream.

#### Layer normalization
The original Transformer also normalizes the output of every residual addition.

For a vector $x\in\mathbb{R}^{d_{\text{model}}}$, layer normalization computes its mean and variance across the feature dimension,
$$
\mu = \frac{1}{d_{\text{model}}} \sum_{j=1}^{d_{\text{model}}} x_j,
$$
as well as
$$
	\sigma^2 = \frac{1}{d_{\text{model}}} \sum_{j=1}^{d_{\text{model}}} (x_j - \mu)^2,
$$
and returns
$$
	\operatorname{LayerNorm}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
$$
:::note $\odot$ is the Hadamard product. ::: where $\gamma$ and $\beta$ are learned parameters.

> Each token representation can be normalized on its own.

The 2017 Transformer uses
$$
	\operatorname{LayerNorm}(x + \operatorname{SubLayer}(x)).
$$
That is, the model performs the sublayer, adds the residual, and then normalizes. This is now generally called **post-norm**.

Later Transformers will reverse the ordering and normalize before the attention or feed-forward operation. We will come back to why.

### The encoder
We can now construct an encoder layer.

Let $H^{(0)}$ be the sequence of token embeddings after positional encodings have been added.

An encoder layer first performs unrestricted self-attention:
$$
	A^{(l)} = \operatorname{LayerNorm}(H^{(l-1)} + \operatorname{MultiHead}(H^{(l-1)}, H^{(l-1)}, H^{(l-1)})).
$$
Then it applies the position-wise feed-forward network:
$$
	H^{(l)} = \operatorname{LayerNorm}(A^{(l)}  + \operatorname{FFN}(A^{(l)})).
$$
The original Transformer repeats this six times.

### The decoder
The decoder is slightly more complicated.

Suppose we are translating
> "I love boxes"
into French.

When predicting the French token at position $t$, the model is allowed to use the source sentence and the French tokens preceding position $t$; it cannot inspect the correct future French tokens. If ordinary self-attention were used during training, however, nothing would stop position $t$ from looking directly at positions $t+1,t+2,\ldots$.

To avoid this pitfall, the authors introduce a *causal mask*.

Before softmax, attention computes the score matrix
$$
	S = \frac{QK^\top}{\sqrt{d_k}}.
$$
We define a mask $M$,
$$
	M_{ij} =
	\begin{cases}
		0, & j \le i\\
		-\infty, & j > i
	\end{cases}
$$
and compute $\operatorname{softmax}(S + M)$. Since $e^{-\infty} = 0$, every future position receives exactly zero attention probability. For four tokens, the causal visibility pattern looks like
$$
	\begin{bmatrix}
		\checkmark & \times & \times & \times \\
		\checkmark & \checkmark & \times & \times \\
		\checkmark & \checkmark & \checkmark & \times \\
		\checkmark & \checkmark & \checkmark & \checkmark
	\end{bmatrix}
$$
Position $1$ can see only itself. Position $2$ can see positions $1$ and $2$. Position $3$ can see positions $1$, $2$, and $3$. And so forth.

#### Cross-attention
After masked self-attention, each decoder layer performs a second attention operation.

This time, the queries come from the decoder, while the keys and values come from the encoder:
$$
	Q = H_{\text{decoder}} W^Q, K = H_{\text{encoder}} W^K, V = H_{\text{encoder}} W^V.
$$
We can interpret this as
> Given what I have generated so far, which parts of the input sentence are relevant to what I should produce next?
Cross-attention is not causally masked over the source sequence. Every decoder position can inspect every encoder position because the entire input sentence is already known. :::note One can imagine a decoder representation corresponding to the next French word attending heavily to the English word or phrase it is currently translating. :::

#### A complete decoder layer
A decoder layer therefore contains three sub-layers.

First, masked self-attention:
$$
	D_1 = \operatorname{LayerNorm}(D + \operatorname{MaskedMultiHead}(D, D, D)).
$$
Second, cross-attention over the encoder output $H$:
$$
	D_2 = \operatorname{LayerNorm}(D_1 + \operatorname{MultiHead}(D_1, H, H)).
$$
Lastly, the feed-forward network:
$$
	D_3 = \operatorname{LayerNorm}(D_2 + \operatorname{FFN}(D_2)).
$$
The base model stacks six of these decoder layers.

### The complete architecture

We can now assemble the entire model. On the encoder side, token embeddings are combined with positional encodings and passed through six identical layers. Each layer applies unrestricted multi-head self-attention followed by a position-wise feed-forward network, with a residual connection and layer normalization around each sublayer.

The decoder receives the target sequence shifted one position to the right, again with positional encodings added. Each of its six layers applies masked self-attention, cross-attention over the encoder output, and a position-wise feed-forward network. Every sublayer is wrapped in the same residual-then-normalize structure.

Finally, a learned linear transformation projects each decoder output into vocabulary-sized logits, and softmax converts those logits into next-token probabilities. During training, the causal mask allows all target positions to be processed in parallel without revealing future tokens. During inference, the decoder still generates one token at a time.

### Why self-attention?
Suppose the sequence has length $n$ and each representation has dimension $d$.

The paper compares the dominant layer types approximately as follows:
| Layer          | Complexity per layer | Sequential operations | Maximum path length |
| -------------- | -------------------: | --------------------: | ------------------: |
| Self-attention |            $O(n^2d)$ |                $O(1)$ |              $O(1)$ |
| Recurrent      |            $O(nd^2)$ |                $O(n)$ |              $O(n)$ |
| Convolutional  |           $O(knd^2)$ |                $O(1)$ |       $O(\log_k n)$ |


One caveat is important: this does not mean autoregressive generation itself suddenly becomes parallel. When producing text, token $t+1$ still depends on token $t$. The Transformer removed sequence-aligned recurrence inside each layer and allowed positions to be processed in parallel during training. Autoregressive decoding remains sequential across generated tokens.

Much of the subsequent history of language-model architecture and systems engineering can be read as an attempt to retain the benefits of global attention while making that $n^2$ less painful: sparse attention, local attention, recurrence-like memory, kernelized approximations, FlashAttention, KV caching, grouped-query attention, sliding windows, and increasingly elaborate serving systems.

We are left with one question:
> **how does it learn anything?**
