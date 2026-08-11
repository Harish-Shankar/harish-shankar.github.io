---
title: The genealogy of language models
date: 2026-08-11
description: How do models actually learn?
timezone: Asia/Kolkata
series: genealogy
series_order: 2
permalink: genealogy/learning
---

## How do models actually learn?
In the previous section, we built the model and hopefully grounded our intuition to understand the choices made and *raison d'être*. However, it should be abundantly clear that there is no information regarding how they learn nor why they learn. The matrices $W^Q,W^K,W^V$ do not begin knowing what a noun is. The embedding for *dog* is not "close" to the embedding for *puppy*. No attention head has been assigned the job of resolving pronouns. The feed-forward networks contain no handwritten rules of grammar.

Surprisingly, the motivation for the model to learn is essentially
> predict the correct token.
For the original transformer, the one created for translation, we add some specificity:
> given the source sentence and the correct target prefix, predict the next target token.
Everything else follows from repeatedly measuring how wrong the model was and changing its parameters so that, next time, the correct token receives slightly more probability, and thus more likely to be chosen. However, before we begin this discussion, it is worth backtracking and clarifying a particular term.

### Tokens
We mentioned previously that the model takes an input sequence of symbol representations $x$. $x$ is not the words that compose the sentence, but instead integers produced by a **tokenizer** that converts text into a sequence drawn from some finite vocabulary
$$
  \mathcal{V} = \{1, \ldots, V\}.
$$
Using our example sentence
> "The transformer unpacks unhelpful words into smaller pieces before processing them."
the [GPT-5.x Tokenizer](https://platform.openai.com/tokenizer) splits it into 15 tokens, namely
:::tokens
The| transformer| un|packs| un|help|ful| words| into| smaller| pieces| before| processing| them|.
:::
whose corresponding token IDs are
$$
  [976, 59595, 537, 103878, 537, 11283, 1500, 6391, 1511, 13679, 12762, 2254, 12323, 1373, 13].
$$
The number of tokens does not match the number of words found nor the number of characters found; tokens are **sub-word**.

#### Why sub-word?
Language is considered to be open vocabulary in that it is not restricted to a predefined set of words or phrases; names, compounds, spelling variations, technical terminology, and newly invented words make it impossible to maintain a reasonably sized vocabulary containing every word one might encounter. Tokenizing every character is untenable as the sequences simply become too long.

Sennrich, Haddow, and Birch :::cite Senntich et al. **Neural Machine Translation of Rare Words with Subword Units** [ARXIV](https://arxiv.org/pdf/1508.07909) ::: had shown that byte-pair encoding could be adapted to neural machine translation by decomposing uncommon words into reusable sub-word units. A frequent word might remain a single token; a rare word might become several smaller tokens. This provided a fixed vocabulary without requiring every possible word to have its own entry.

A larger vocabulary allows more strings to be represented with a single token, shortening sequences, but requires larger embedding and output matrices. A smaller vocabulary reduces those matrices but produces longer sequences. And because ordinary self-attention scales quadratically with sequence length, tokenization ultimately affects far more than vocabulary size.

Tokenization will eventually become an engineering problem of its own. For now, assume that some tokenizer has handed us a sequence of token IDs.

### Embeddings
Looking back out our sample tokenized sentence and their corresponding token IDs, we see
$$
  \textit{ful} \rightarrow 1500 \quad \text{and} \quad \textit{ into} \rightarrow 1511 .
$$
In relation to the vocabulary size, both of these sub-words are close to each other, yet there is no obvious relation between each morpheme. :::definition A **morpheme** is the smallest unit of meaning in a language. It cannot be broken into smaller parts without losing its meaning. :::

The fact that two tokens have nearby IDs tells us nothing about whether the corresponding pieces of language are similar. Before the Transformer can operate on tokens, we therefore need to place them into a continuous vector space. Let
$$
  E \in \mathbb{R}^{V \times d_{\text{model}}}
$$
be the embedding matrix.

Each row corresponds to one token in the vocabulary. If the token at position $t$ is $x_t$, its embedding is simply
$$
  e_t = E_{x_t}.
$$

$E$ is a learned parameter matrix.

At initialization, the vector associated with *dog* has no particular reason to resemble the vector associated with *puppy*. If they eventually become related, it is because using similar representations for them happened to make the model better at its training objective: **predicting the correct token**.

In Vaswani et al.'s original paper, :::note They specify this scaling but do not give a derivation for it. ::: the embedding is multiplied by $\sqrt{d_{\text{model}}}$ before the positional (encoding) information is added. Thus, the initial representation is
$$
  h_t^{(0)} = \sqrt{d_{\text{model}}}E_{x_t} + \operatorname{PE}_t.
$$

We now have vectors entering the Transformer. Though, eventually, we need to turn vectors back into tokens.

### Logits
Before proceeding, a word on notation. The embedding matrix $E$ and the positional encoding $\operatorname{PE}$ act identically on the source and the target; nothing in the previous section had to distinguish them. From here on we need to speak of both at once, so let
$$
  x = (x_1, \ldots, x_n) \quad \text{and} \quad y = (y_1, \ldots, y_m)
$$
denote the source and target sequences respectively, both drawn from $\mathcal{V}$.

Suppose the decoder has consumed the source $x$ together with the target prefix $y_{\leq t}$. Because of the causal mask, the representation produced by the final decoder layer at position $t$ sees the whole of the source but nothing of the target beyond $y_t$. We make that dependence explicit and write
$$
  h_t = f_\theta^{(t)}\!\left(y_{\leq t}, x\right) \in \mathbb{R}^{d_{\text{model}}},
$$
where $f_\theta^{(t)}$ is the composition of every decoder layer evaluated at position $t$, and $\theta$ collects every parameter of the model. Now the model must answer:
> which token should come next?

We assign every possible token a score. If
$$
  W_{\text{out}} \in \mathbb{R}^{V \times d_{\text{model}}},
$$
then,
$$
  \ell_t = W_{\text{out}}h_t \in \mathbb{R}^{V}.
$$
The vector $\ell_t$ contains one scalar for every vocabulary item. These numbers are called logits.

Softmax converts them into a distribution over $\mathcal{V}$:
$$
  p_\theta\!\left(y_{t+1} = v \mid y_{\leq t}, x\right)
  = \operatorname{softmax}(\ell_t)_v
  = \frac{\exp(\ell_{t,v})}{\sum_{u \in \mathcal{V}} \exp(\ell_{t,u})},
  \qquad v \in \mathcal{V}.
$$
Writing $p_t = \operatorname{softmax}(\ell_t)$, we have $p_{t,v} \geq 0$ for every $v$ and $\sum_{v \in \mathcal{V}} p_{t,v} = 1$. :::note Equivalently, , $p_t \in \Delta^{V-1}$, the probability simplex in $\mathbb{R}^V$. ::: If our vocabulary contains $37{,}000$ tokens, the model emits $37{,}000$ probabilities at every prediction position.

Two remarks before we continue.

Applying the chain rule to the target sequence gives
$$
  p_\theta(y \mid x) = \prod_{t=0}^{m-1} p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right),
$$
where $y_{\leq 0}$ is the empty prefix. :::note In practice the empty prefix is realized as a dedicated start-of-sequence token, so that position $1$ has something to attend to. ::: Predicting the correct token *is* maximizing the likelihood of the whole target sentence.

Softmax is also invariant to a common shift. For any $c \in \mathbb{R}$,
$$
  \operatorname{softmax}(\ell_t + c\mathbf{1}) = \operatorname{softmax}(\ell_t),
$$
so logits carry meaning only up to an additive constant. Their absolute magnitudes are not identifiable; only their differences are. :::note This invariance is what licenses the standard implementation, which subtracts $\max_{u} \ell_{t,u}$ before exponentiating and thereby avoids overflow. :::

One might think that the embedding matrix used to read tokens and the output matrix used to predict tokens should be entirely unrelated.

Curiously, they need not be.
:::progress 2026-08-11T05:38:42.094Z