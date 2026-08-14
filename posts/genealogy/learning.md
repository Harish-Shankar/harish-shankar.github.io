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
In the previous section, we built the model and hopefully grounded our intuition to understand the choices made and their *raison d'être*. What should be abundantly clear, however, is that nothing in that construction tells us how the model learns, nor why it learns. Nothing in that construction knows any language. $W^Q,W^K,W^V$ do not begin knowing what a noun is. The embedding for *dog* is not close to the embedding for *puppy*. No attention head has been assigned the job of resolving pronouns. The feed-forward networks contain no handwritten rules of grammar.

Surprisingly, the motivation for the model to learn is essentially
> predict the correct token.
For the original Transformer, the one created for translation, we add some specificity:
> given the source sentence and the correct target prefix, predict the next target token.
Everything else follows from repeatedly measuring how wrong the model was and changing its parameters so that, next time, the correct token receives slightly more probability, and is thus more likely to be chosen. Before we begin this discussion, however, it is worth backtracking and clarifying a term I have been using loosely for an entire post.

### Tokens
We mentioned previously that the model takes an input sequence of symbol representations $x$. $x$ is not the words that compose the sentence, but instead integers produced by a **tokenizer** that converts text into a sequence drawn from some finite vocabulary
$$
  \mathcal{V} = \{1, \ldots, V\}.
$$
Using an example sentence
> "The transformer unpacks unhelpful words into smaller pieces before processing them."
the [GPT-5.x Tokenizer](https://platform.openai.com/tokenizer) splits it into 15 tokens, namely
:::tokens
The| transformer| un|packs| un|help|ful| words| into| smaller| pieces| before| processing| them|.
:::
whose corresponding token IDs are
$$
  [976, 59595, 537, 103878, 537, 11283, 1500, 6391, 1511, 13679, 12762, 2254, 12323, 1373, 13].
$$
The number of tokens matches neither the number of words nor the number of characters; tokens are **sub-word**.

#### Why sub-word?
Language is open vocabulary; in that it is not restricted to a predefined set of words or phrases; names, compounds, spelling variations, technical terminology, and newly invented words make it impossible to maintain a reasonably sized vocabulary containing every word one might encounter. So, unless we are willing to constantly document and tokenize every new word and its spelling variations from all across the globe, it is just untenable as tokenizing approach.

Character-level tokenization also doesn't make much sense. The sentence above is 83 characters and 15 tokens. Since ordinary self-attention scales quadratically with sequence length, a character-level tokenizer would ask for roughly thirty times the attention work to say the same thing.

Sennrich, Haddow, and Birch :::cite Sennrich et al. **Neural Machine Translation of Rare Words with Subword Units.** 2015. [PDF](https://arxiv.org/pdf/1508.07909) ::: had shown that byte-pair encoding could be adapted to neural machine translation by decomposing uncommon words into reusable sub-word units. A frequent word might remain a single token; a rare word might become several smaller tokens. This provided a fixed vocabulary without requiring every possible word to have its own entry.

The remaining choice is where to sit between the two failures. A larger vocabulary represents more strings per token and shortens sequences, but enlarges the embedding and output matrices. A smaller vocabulary shrinks those matrices and lengthens sequences, and length is the expensive axis.

Sub-word tokenization is a compromise nobody would have designed on purpose, and it is still what everything runs on. It will eventually become an engineering problem of its own, and I suspect a post of its own. For now, assume some tokenizer has handed us a sequence of token IDs and has been thanked for its trouble.

### Embeddings
Looking back at our tokenized sentence and its IDs, we see
$$
  \textit{ful} \rightarrow 1500 \quad \text{and} \quad \textit{ into} \rightarrow 1511 .
$$
Relative to the size of the vocabulary these two IDs are neighbors, and there is no relation whatsoever between the two morphemes. :::definition A **morpheme** is the smallest unit of meaning in a language. It cannot be broken into smaller parts without losing its meaning. :::

Proximity in ID space carries no information. Before the Transformer can operate on tokens we therefore have to place them into a continuous space where proximity can mean something. Let
$$
  E \in \mathbb{R}^{V \times d_{\text{model}}}
$$
be the embedding matrix. Each row corresponds to one token in the vocabulary. If the token at position $t$ is $x_t$, its embedding is
$$
  e_t = E_{x_t}.
$$
Calling this a *lookup* makes it sound as though some semantic dictionary has been handed to the model. It has not. $E$ is a **learned parameter matrix**. At initialization, the vector associated with *dog* has no particular reason to resemble the vector associated with *puppy*. If they eventually come to resemble each other, it is because using similar representations for them happened to make the model better at predicting the correct token.

In Vaswani et al., the embedding is multiplied by $\sqrt{d_{\text{model}}}$ before positional information is added, so the initial representation is
$$
  h_t^{(0)} = \sqrt{d_{\text{model}}}E_{x_t} + \operatorname{PE}_t.
$$
The paper states the scaling and does not derive it, which is the first of several places where Vaswani et al. report a constant bluntly. The usual reconstruction is scale matching. Embedding entries are typically initialized with standard deviation around $1/\sqrt{d_{\text{model}}}$, so each embedding vector has norm around $1$; the sinusoidal positional encodings have entries bounded by $1$ and therefore a norm around $\sqrt{d_{\text{model}}/2}$, which for $d_{\text{model}} = 512$ is about $16$. Added directly, position would drown out identity. Multiplying the embedding by $\sqrt{d_{\text{model}}} \approx 22.6$ puts the two on comparable footing. :::note This argument becomes load-bearing in the next section, where the same matrix is asked to serve as the output projection, and its rows therefore also have to produce logits of a sensible magnitude. :::

We now have vectors entering the Transformer. Eventually we need to turn vectors back into tokens.

### Logits
Before proceeding, a word on notation. The embedding matrix $E$ and the positional encoding $\operatorname{PE}$ act identically on the source and the target, so nothing in the previous section had to distinguish them. From here on we need to speak of both at once, so let
$$
  x = (x_1, \ldots, x_n) \quad \text{and} \quad y = (y_1, \ldots, y_m)
$$
denote the source and target sequences respectively, both drawn from $\mathcal{V}$.

Suppose the decoder has consumed the source $x$ together with the target prefix $y_{\leq t}$. Because of the causal mask, the representation produced by the final decoder layer at position $t$ sees the whole of the source but nothing of the target beyond $y_t$. Making that dependence explicit,
$$
  h_t = f_\theta^{(t)}\!\left(y_{\leq t}, x\right) \in \mathbb{R}^{d_{\text{model}}},
$$
where $f_\theta^{(t)}$ is the composition of every decoder layer evaluated at position $t$, and $\theta$ collects every parameter of the model. Now the model must answer:
> which token should come next?

We assign every possible token a score. If
$$
  W_{\text{out}} \in \mathbb{R}^{V \times d_{\text{model}}},
$$
then
$$
  \ell_t = W_{\text{out}}h_t \in \mathbb{R}^{V}.
$$
The vector $\ell_t$ contains one scalar for every vocabulary item. These numbers are the logits. Softmax converts them into a distribution over $\mathcal{V}$:
$$
  p_\theta\!\left(y_{t+1} = v \mid y_{\leq t}, x\right)
  = \operatorname{softmax}(\ell_t)_v
  = \frac{\exp(\ell_{t,v})}{\sum_{u \in \mathcal{V}} \exp(\ell_{t,u})},
  \qquad v \in \mathcal{V}.
$$
Writing $p_t = \operatorname{softmax}(\ell_t)$, we have $p_{t,v} \geq 0$ for every $v$ and $\sum_{v \in \mathcal{V}} p_{t,v} = 1$. :::note Equivalently, $p_t \in \Delta^{V-1}$, the probability simplex in $\mathbb{R}^V$. ::: With $37{,}000$ tokens in the vocabulary, the model emits $37{,}000$ probabilities at every prediction position, of which exactly one will be graded.

Two properties of this construction matter later, though it is worth explicitly talking about them now.

The first is that the chain rule applies to the target sequence:
$$
  p_\theta(y \mid x) = \prod_{t=0}^{m-1} p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right),
$$
where $y_{\leq 0}$ is the empty prefix. :::note In practice the empty prefix is realized as a dedicated start-of-sequence token, so that position $1$ has something to attend to. ::: Predicting the correct token *is* maximizing the likelihood of the whole target sentence. The local objective and the global one are the same objective, exactly, with no approximation in between.

The second is that softmax is invariant to a common shift. For any $c \in \mathbb{R}$,
$$
  \operatorname{softmax}(\ell_t + c\mathbf{1}) = \operatorname{softmax}(\ell_t),
$$
so logits carry meaning only up to an additive constant. Their absolute magnitudes are not identifiable; only their differences are. :::note This invariance is what licenses the standard implementation, which subtracts $\max_{u} \ell_{t,u}$ before exponentiating and thereby avoids overflow. :::

One might think that the matrix used to read tokens and the matrix used to predict them should be entirely unrelated. They need not be!
:::progress 2026-08-11T05:38:42.094Z

### Weight tying
As described, we have an embedding matrix
$$
  E \in \mathbb{R}^{V \times d_{\text{model}}}
$$
that maps tokens to vectors, and an "output" matrix
$$
  W_{\text{out}} \in \mathbb{R}^{V \times d_{\text{model}}}
$$
that scores hidden vectors against possible output tokens. With $V = 37{,}000$ and $d_{\text{model}} = 512$, each is about $19$ million parameters. The base model has $65$ million in total, so keeping two independent copies would take the model to roughly $84$ million and place close to half of it in two eventual learned "lookup" tables.

This, I believe, is more than enough to consider grouping these matrices into one. With that being said, there is also a conceptual argument that has enough merit and is the reason I am talking about this in the first place. I ask you, what does a row in mean in these matrices? The row $E_v$ answers
> what vector should represent token $v$ when I read it?
while $(W_{\text{out}})_v$ answers
> what direction in representation space should make me believe the answer is token $v$?
Press and Wolf :::cite Ofir Press, Lior Wolf. **Using the Output Embedding to Improve Language Models**. 2016. [PDF](https://arxiv.org/pdf/1608.05859) ::: showed that the output matrix of a neural language model already behaves as a meaningful word embedding in its own right, and that tying the input and output matrices could improve perplexity while substantially reducing parameter count. In neural translation models their experiments found that tying reduced model size dramatically without degrading performance. Inan, Khosravi, and Socher arrived independently at closely related weight-sharing results from a loss-based analysis. :::cite Inan, Khosravi, and Socher. **Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling.** 2016. [PDF](https://arxiv.org/pdf/1611.01462) :::

We therefore impose
$$
  W_{\text{out}} = E,
$$
and with our convention that vocabulary embeddings occupy the rows of $E$, logits become
$$
  \ell_t = Eh_t.
$$
Read that way, a logit is an inner product between the current hidden state and the embedding of a candidate token. Prediction becomes a similarity search in the same space that reading uses. The original Transformer goes one step further and shares a single matrix across both embedding layers and the pre-softmax transformation, which is commonly called three-way weight tying.

Weight tying reduces capacity in exchange for parameter efficiency and an inductive bias that says these two problems ought to share structure. There is no mathematical necessity that the best space for interpreting an input token is the best space for classifying an output token. Tying asserts that it is, on the strength of a plausible story and a better perplexity number.

We can now read symbols, transform them, and turn the final representations into probabilities. We have no definition of being wrong.

### The objective
Let $x = \langle x_1, \ldots, x_n \rangle$ be the source sentence and $y = \langle y_1, \ldots, y_m \rangle$ be the target translation. We want the model to assign high probability to the entire correct translation, $p_\theta(y \mid x)$, and by the factorization above that is the product of the probabilities assigned to each correct next token:
$$
  p(y \mid x) = p(y_1 \mid x) \cdot p(y_2 \mid y_1, x) \cdot p(y_3 \mid y_2, y_1, x) \cdot \dots.
$$
Products of many small probabilities underflow. A target of $100$ tokens each predicted at probability $0.1$ has joint probability $10^{-100}$, which single-precision floating point cannot represent at all. Sums of logarithms have no such problem, so we work in log-space:
$$
  \log p_\theta(y \mid x) = \sum_{t=0}^{m-1} \log p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right).
$$
Maximum likelihood estimation chooses the parameters that maximize this quantity over the training set:
$$
  \theta^\ast = \arg\max_{\theta} \sum_{(x,y) \in \mathcal{D}}\sum_{t=0}^{|y|-1} \log p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right).
$$
Optimization libraries conventionally minimize rather than maximize, so we negate:
$$
  \mathcal{L}_{\text{NLL}} = -\sum_{(x,y) \in \mathcal{D}}\sum_{t=0}^{|y|-1} \log p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right).
$$
This is the **negative log-likelihood**, and for a one-hot target distribution it is exactly the usual cross-entropy loss.

It is worth noting for our discussions down-the-line that if we remove the encoder and the source sentence $x$, the objective becomes
$$
  p_\theta(x_1,\ldots,x_n) = \prod_{t=1}^n p_\theta(x_t \mid x_{< t}),
$$
with loss
$$
  \mathcal{L} = - \sum_t \log p_\theta\!\left(x_{t} \mid x_{< t}\right).
$$
That is the loss that will eventually train most of the models in this series.

#### Cross-entropy
Suppose the correct next token is $y$, and let $q$ be its one-hot target distribution:
$$
  q_v =
  \begin{cases}
    1, & v = y,\\
    0, & v \ne y.
  \end{cases}
$$
The cross-entropy between the target distribution $q$ and model distribution $p$ is
$$
  H(q,p) = -\sum_{v=1}^V q_v \log p_v = -\log p_y
$$
because only one entry of $q$ is nonzero. The entire vocabulary disappears from the written expression: the loss is simply the negative log-probability assigned to the correct answer.

Some values, to make the scale concrete. A model that assigns the correct token $p_y = 0.9$ incurs about $0.105$. At $p_y = 0.01$ it incurs about $4.605$. A uniform guess over $37{,}000$ tokens incurs $10.5$, which is where training starts. It seems obvious that confident wrong answers are hampered more severely.

Let $\ell_j$ denote the logit for vocabulary item $j$. Combining softmax and cross-entropy gives
$$
  \frac{\partial \mathcal{L}}{\partial \ell_j} = p_j - q_j.
$$
For the correct token this is $p_y - 1$, negative unless the model already assigns probability $1$, so gradient descent pushes that logit up. For every incorrect token $p_j - q_j > 0$, so it pushes down, and by more for the alternatives the model found most tempting.

Two things to keep in the back of your mind:
1. The magnitude is bounded. However catastrophically wrong the model is, no logit gradient exceeds $1$ in absolute value. The loss itself is unbounded above, but the signal it emits is not.
2. The gradient sums to zero: $\sum_j (p_j - q_j) = 1 - 1 = 0$. Probability at this layer is conserved, so the correct token cannot be raised without something else being lowered. Learning here is redistribution.

#### Causal shifting
Suppose the target sequence is $y_1, y_2, y_3, y_4$. To train
$$
  p(y_3 \mid y_2,y_1,x),
$$
the model must be shown $y_1$ and $y_2$ but not $y_3$. To train
$$
  p(y_4 \mid y_3,y_2,y_1,x),
$$
it must see $y_3$. One naive solution is to run the model once to predict $y_1$, again for $y_2$, again for $y_3$, and so on, which throws away the parallelism we built the Transformer to get. Instead we shift the target sequence right.

Suppose the correct translation is :::note $\langle\mathrm{BOS}\rangle$ is a special token that represents the beginning of a sequence, while $\langle\mathrm{EOS}\rangle$ is a special token that represents the end of a sequence. :::
$$
  y_1,y_2,y_3,\langle\mathrm{EOS}\rangle.
$$
The decoder receives
| Position | Decoder input | Prediction target |
| -------- | ------------- | ----------------- |
| 1        | `<BOS>`       | $y_1$             |
| 2        | $y_1$         | $y_2$             |
| 3        | $y_2$         | $y_3$             |
| 4        | $y_3$         | `<EOS>`           |

Equivalently,
$$
  \text{decoder input} =  \langle\mathrm{BOS}\rangle, y_1, y_2, y_3,
$$
while
$$
  \text{targets} = \{y_1, y_2, y_3, \langle\mathrm{EOS}\rangle\}.
$$

The causal mask prevents position $t$ from looking to its right, so
1. position $1$ sees `<BOS>` and predicts $y_1$;
2. position $2$ sees `<BOS>`, $y_1$ and predicts $y_2$;
3. position $3$ sees `<BOS>`, $y_1,$ $y_2$ and predicts $y_3$;
4. position $4$ sees `<BOS>`, $y_1,$ $y_2,$ $y_3$ and predicts `<EOS>`.
Every position's correct prefix is already available, so all four predictions are computed in a single forward pass
 > Autoregressive does not mean that training must itself proceed one token at a time.

Generation will be sequential. Training does not have to be.

#### Teacher forcing
The construction above quietly made a choice worth discussing: teacher forcing, which feeds the *true* previous value rather than the network's own previous output back into the computation during training. :::cite Williams and Zipser. **A Learning Algorithm for Continually Running Fully Recurrent Neural Networks.** 1989. [PDF](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf) ::: With an example, suppose the target begins
> The transformer unpacks...
When learning to predict *unpacks*, we condition the model on the correct prefix
> The transformer
regardless of whether the model itself would have predicted *transformer* one step earlier.

Consider the alternative. An untrained model produces nonsense. Feed that nonsense back as context and later predictions are conditioned on increasingly meaningless prefixes, so the model spends its first several thousand steps learning the correct continuations of sequences that appear nowhere in any language. Teacher forcing keeps the trajectory anchored to real examples, at a price we are about to pay.

It also combines particularly well with causal masking. Since every correct previous token is known, the entire shifted target can enter the decoder at once, and a sentence containing $m$ yields roughly $m$ supervised next-token problems for the price of one forward pass.
:::progress 2026-08-11T11:45:36.271Z

### Training & Inference
During training, when predicting token $t$, the model receives $y_{< t}$, the correct prefix, irrespective of what it would have chosen. During inference :::definition **Inference** is the process of using an already trained large language model to generate text or answers for new user prompts. ::: it receives $\hat y_{< t}$, its own generated prefix, which may be wrong.

This train-test mismatch became known as **exposure bias**. Bengio et al. :::cite Bengio et al. **Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.** 2015. [PDF](https://arxiv.org/pdf/1506.03099) ::: proposed scheduled sampling: gradually replace some ground-truth previous tokens during training with tokens generated by the model itself, thereby exposing the model to the kinds of prefixes it will encounter at inference.

Huszár then showed that the scheduled-sampling objective is not, in general, a consistent estimator of the data distribution. :::cite Huszár. **How (not) to Train your Generative Model.** 2015. [PDF](https://arxiv.org/pdf/1511.05101) ::: The fix changes which distribution the model is incentivized to learn.

:::remark Being pedantic
It is common to say "teacher forcing causes exposure bias, therefore maximum likelihood is flawed."

The autoregressive factorization is exact. If every learned conditional matched the true conditional distribution, sampling autoregressively from the model would reproduce the correct joint distribution.

The practical problem is that our learned conditionals are imperfect. During rollout, an imperfect prediction may place us in regions of sequence space where those approximations are worse, allowing errors to compound.

So there is a genuine train-generation mismatch, but simply replacing maximum likelihood with model-generated prefixes does not automatically solve it.
:::
This returns much later under other names and other training paradigms. For the 2017 Transformer we stick with *teacher-forced maximum likelihood*.

#### How does loss reach attention heads?
Return to the example sentence from the previous section, "Because the trophy was too large for the suitcase, it did not fit." Suppose one attention head eventually learns that *it* should retrieve information from *trophy*. Who told it to? Nobody. The only supervision available was that, several layers later, the model assigned insufficient probability to the correct translated token. **Backpropagation** is what carries that fact backwards. :::definition **Backpropagation** is the repeated application of the chain rule over a computation graph, computing the gradient of a scalar loss with respect to every intermediate value and parameter in a single backward pass. :::

Consider a single attention operation $Z = AV$, where
$$
  A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),
$$
and suppose the gradient arriving from later layers is
$$
  G = \frac{\partial \mathcal{L}}{\partial Z}.
$$
Then
$$
  \frac{\partial \mathcal L}{\partial V} = A^\top G,
$$
so the forward attention weights also determine where learning signal is routed backward through the value pathway. A position that was ignored in the forward pass receives almost no gradient. Similarly,
$$
  \frac{\partial \mathcal L}{\partial A} = GV^\top.
$$
The $(i,j)$ entry is $\langle g_i, v_j \rangle$: the inner product between the direction position $i$'s output should have moved and the content sitting at position $j$. If the information at $j$ points the way the loss wanted to go, the gradient asks for more attention on $j$. Credit assignment across positions is itself a similarity computation, over the same vectors the forward pass used.

That error then propagates backward through the softmax and through $QK^\top$, producing gradients for $Q$ and $K$ and ultimately for $W^Q$, $W^K$, and $W^V$. Since those were themselves computed from earlier hidden representations, the gradient continues through every preceding block to the embeddings. The whole path:
1. **The error appears at the output.** At each target position the loss measures how much probability the model failed to assign to the correct next token.
2. **We differentiate through the softmax,** starting from $\partial \mathcal L / \partial \ell_j = p_j - q_j$.
3. **We pass through the output projection.** Since $\ell_t = W_{\text{out}}h_t$, the logit gradients say how the final hidden state should have differed. Under weight tying, this step also contributes to the gradient of the shared matrix.
4. **We pass through the decoder block.** Feed-forward network, residual connections, layer normalization, cross-attention, and masked self-attention each receive a signal describing how changing their output would have changed the loss.
5. **Attention splits the signal in two.** Gradients flow through the retrieved content $V$ and through the weights $A$, so the model learns both what to carry and where to look.
6. **The projections are updated.** If making *it* attend more strongly to *trophy* would have raised the probability of the correct output token, the parameters governing that query-key compatibility move in exactly that direction.
7. **The gradient keeps going,** recursively through earlier blocks and into the embeddings, so tokens useful in similar contexts can, over many examples, acquire related structure.

What we are left with is $\nabla_\theta \mathcal L$: a locally valid description of how the loss would change if each parameter were perturbed. No attention head was ever told that this pronoun refers to *trophy*. A single scalar at the end of the network became millions of local redistribution signals.
 
Backpropagation gives the direction in which each parameter affects the loss. It says nothing at all about how far to move.

### How does it improve?
We do not update the network after a single token, nor after a single sentence.

Training uses minibatches. The English-German experiment used about $4.5$ million sentence pairs and the English-French dataset about $36$ million. Sentence pairs of similar length were grouped together, with each batch holding roughly $25{,}000$ source tokens and $25{,}000$ target tokens. :::note Sequences in a batch need compatible tensor shapes, so shorter ones are padded; a batch containing lengths 10, 12, 15 and 500 pads the short examples nearly to 500 and wastes an enormous amount of computation. Batching similar lengths avoids paying for padding. :::

The paper describes batch size in tokens, not in sentences. A batch of 100 ten-token sequences has very different computational cost from a batch of 100 thousand-token sequences. This distinction will eventually become standard operating procedure for large-scale training.

For each minibatch we obtain a scalar loss $\mathcal L(\theta)$, differentiate it,
$$
  g_t = \nabla_\theta \mathcal L(\theta_t),
$$
and decide how far to move. Naively,
$$
  \theta_{t+1} = \theta_t - \eta g_t,
$$
which is **gradient descent**. Its problem is that $\eta$ is shared by every parameter in the model, and the parameters do not agree about scale.

#### Adam
Suppose one parameter repeatedly receives gradients like $\{0.0001, 0.0002, 0.0002, \ldots \}$ while another receives $\{8, -1, 4, \ldots \}$. Any single $\eta$ small enough to keep the second stable leaves the first effectively frozen. Adam (adaptive moment estimation) :::cite Kingma and Ba. **Adam: A Method for Stochastic Optimization.** 2014. [PDF](https://arxiv.org/pdf/1412.6980) ::: fixes this by maintaining running statistics per parameter.

Given gradient $g_t$, Adam tracks an exponential moving average of the gradient,
$$
  m_t = \beta_1m_{t-1} + (1-\beta_1)g_t,
$$
an estimate of the first moment. Informally, $m_t$ behaves like momentum: it smooths the noisy minibatch gradient and preserves recent direction. It also tracks an exponential moving average of the squared gradient,
$$
  v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2,
$$
an estimate of the second raw moment. Both averages begin at zero, which biases early estimates downward, so Adam corrects them:
$$
  \hat m_t = \frac{m_t}{1-\beta_1^t} \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}.
$$
The update is
$$
  \theta_t = \theta_{t-1} - \eta_t \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
$$
:::progress 2026-08-12T06:15:31.114Z

A parameter that consistently receives large-magnitude gradients accumulates a large $\hat v_t$, and its effective step is damped accordingly; a parameter with persistently small gradients is scaled up. 

:::remark THe diameter is the whole point
  Replace $g$ with $cg$ for some $c>0$. Then,
  $$
    m \mapsto cm, \quad v \mapsto c^2v, \quad \sqrt{v} \mapsto c\sqrt{v},
  $$
  and the ratio
  $$
    \frac{\hat m_t}{\sqrt{\hat v_t}}
  $$
  does not move.
:::
The update is scale-invariant in the gradient. :::note $\epsilon$ is doing no statistical work here. It exists so that a parameter whose gradients have been essentially zero does not divide by zero and receive an enormous update. :::

Run our two example parameters through it. The first has gradients around $2 \times 10^{-4}$, so $\sqrt{\hat v} \approx 2 \times 10^{-4}$ and the ratio is about $1$. The second has gradients around $4$, so $\sqrt{\hat v} \approx 4$ and the ratio is again about $1$. Both parameters move by roughly $\eta$, despite gradients four orders of magnitude apart.

Adam is not scaling gradient magnitude down. It is discarding gradient magnitude almost entirely and keeping direction and consistency. A parameter whose gradients agree with each other gets a ratio near $1$; a parameter whose gradients keep cancelling gets a small numerator over a large denominator and barely moves. What $\eta$ now controls is a step size in parameter space.

Kingma and Ba suggested the defaults
$$
  \beta_1 = 0.9, \qquad \beta_2 = 0.999, \qquad \epsilon = 10^{-8},
$$
whereas the Transformer used
$$
  \beta_1 = 0.9, \qquad \beta_2 = 0.98, \qquad \epsilon = 10^{-9}.
$$
The lower $\beta_2$ places less weight on distant history and reacts faster to recent gradient magnitudes; with $\beta_2 = 0.999$ the variance estimate has a memory of roughly a thousand steps, and at $0.98$ roughly fifty. Vaswani et al. report the choice and say nothing else about it.

So Adam converts a gradient into a direction and a relative step. The absolute step, $\eta_t$, is still free, and here the paper does something peculiar

#### Warmup
A learning rate determines how aggressively we trust the current gradient. Too small, and learning proceeds painfully slowly. Too large, and we overshoot useful regions of parameter space or destabilize optimization entirely.

The Transformer does not hold it constant. It uses
$$
  \eta_t = d_{\text{model}}^{-1/2} \min\left(t^{-1/2}, \; t \cdot w^{-3/2}\right),
$$
where $w=4000$ is the number of **warmup steps**. :::note The base model trained for $100{,}000$ steps, so warmup occupies roughly the first four percent of training. :::

While $t < w$, the second term is smaller, so
$$
  \eta_t = d_{\text{model}}^{-1/2}\, t \cdot w^{-3/2},
$$
and the learning rate rises linearly with the step count. At $t = w$ the branches meet, since $w^{-1/2} = w \cdot w^{-3/2}$, and the rate peaks. Thereafter
$$
  \eta_t = d_{\text{model}}^{-1/2}t^{-1/2},
$$
decaying with the inverse square root of the step. Putting $d_{\text{model}} = 512$ into those expressions: the peak at step $4000$ is about $7 \times 10^{-4}$, and by step $100{,}000$ the rate has fallen to about $1.4 \times 10^{-4}$. *Begin cautiously and grow the updates; once learning is established, shrink them*. The $d_{\text{model}}^{-1/2}$ factor additionally makes the whole schedule smaller as the model gets wider.

Why warm up at all? The paper offers the schedule as *obvious* and derives none of it, but later work supplies an illuminating piece of hindsight. Recall from the previous section that the original Transformer normalizes *after* the residual addition,
$$
  \operatorname{LayerNorm}(x + \operatorname{SubLayer}(x)),
$$
and that I promised to explain why later architectures reverse this. Xiong et al. :::cite Xiong et al. **On Layer Normalization in the Transformer Architecture.** 2020. [PDF](https://arxiv.org/abs/2002.04745) ::: analyzed Transformer optimization and argued that post-norm Transformers can have unusually large gradients near the output layers at initialization. Applying a large learning rate immediately to those gradients destabilizes training; warmup buys an initial period of small, cautious updates instead. The same analysis explains why pre-norm Transformers are substantially easier to optimize, and why warmup stopped being load-bearing once the normalization moved.

Which is to say: the schedule was compensating for the architecture.

### Dropout
A sufficiently expressive network can lower training loss by learning peculiarities of the training set that do not generalize. Dropout attacks this by making the network unreliable to itself during training. :::definition **Regularization** is any modification to a learning procedure intended to reduce generalization error rather than training error. :::

For an activation vector $h$, sample a binary mask
$$
  m_i \sim \operatorname{Bernoulli}(1-p),
$$
and, under the commonly used inverted-dropout convention, compute
$$
  \tilde h = \frac{m \odot h}{1-p}.
$$
:::note Recall from the previous section that $\odot$ is the Hadamard product. ::: With probability $p$, an activation is set to zero. The factor $1/(1-p)$ keeps the expected activation unchanged, $\mathbb{E}[\tilde h_i] = h_i$. :::note This is the point of the inverted convention: because the expectation is preserved during training, the network can be run deterministically at inference without rescaling anything. ::: The network consequently cannot depend too heavily on some exact collection of activations always being present. Srivastava et al. :::cite Srivastava et al. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting.** 2014. [PDF](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) ::: motivated this partly in terms of discouraging excessive co-adaptation between features, and partly as an efficient approximation to averaging many different thinned networks.

In the original Transformer, dropout is applied to each sublayer's output *before* the residual addition and normalization,
$$
  \operatorname{LayerNorm}(x + \operatorname{Dropout}(\operatorname{SubLayer}(x))),
$$
and again to the sum of the token embeddings and positional encodings. The base model uses $p_{\text{drop}} = 0.1$, and the paper's ablations show that dropping it hurt.

Here is another tradeoff. Dropout injects noise into optimization, so training on any individual batch becomes a worse approximation of what the full deterministic model would have produced. In exchange, the network is discouraged from relying on fragile co-adaptations that only succeed on its training examples. Regularization is, in a sense, deliberately making the training problem harder in the hope that the solution becomes more general.

The Transformer employs one more, rather more interesting, instance of this philosophy.

### Label smoothing
The cross-entropy target so far has been brutally certain. If token $y$ is correct, then $q_y = 1$ and $q_j = 0$ for every $j \ne y$, which asserts that the training data contains no uncertainty whatsoever. The model is rewarded for driving $p_y \rightarrow 1$ and every $p_j \rightarrow 0$, and at no point is it told that being $95\%$ confident would have been quite enough.

Contextualize this point with regards to the motivating problem for the Transformer architecture: translation. A translated sentence has several acceptable words to choose from that retain the meaning of the original sentence. The one-hot target insists the other words are worth nothing. We have been training the model to be certain about something that isn't and likely will not be certain.

Szegedy et al. :::cite Szegedy et al. **Rethinking the Inception Architecture for Computer Vision.** 2015. [PDF](https://arxiv.org/pdf/1512.00567) ::: introduced **label smoothing** to discourage that extreme. Rather than train against the one-hot $q$, we mix it with a background distribution $u$:
$$
  q' = (1-\epsilon)q + \epsilon u.
$$
For a uniform background over $V$ classes, $u_j = 1/V$, so
$$
  q'_j = (1-\epsilon)q_j + \frac{\epsilon}{V}.
$$
The correct token therefore receives $1 - \epsilon + \epsilon/V$ and every incorrect token receives $\epsilon/V$. The Transformer used $\epsilon_{\text{ls}} = 0.1$. :::note With $V = 37{,}000$, that leaves the correct token a target of roughly $0.9$ and spreads the remaining tenth of the mass across the vocabulary at about $2.7 \times 10^{-6}$ each. :::

The loss is still cross-entropy,
$$
  \mathcal{L} = -\sum_j q'_j \log p_j,
$$
and the gradient becomes
$$
  \frac{\partial \mathcal{L}}{\partial \ell_j} = p_j - q'_j.
$$
The optimization target is no longer a delta distribution—the model is now actively penalized for being too certain. One consequence is easy to overlook: the loss no longer bottoms out at zero. A model that predicted $q'$ perfectly would still pay $H(q') \approx 1.38$ nats :::definition Nats are a unit for measuring information or entropy when logarithms use base $e$. Evaluating $- \ln p$ is how many nats. :::, nearly all of it from the tenth of the mass scattered across $37{,}000$ tokens that no model can ever place correctly.

Label smoothing worsened perplexity while improving accuracy and BLEU. :::definition **BLEU** scores a candidate translation by comparing its $n$-grams against one or more reference translations, with a penalty for being too short. It is a sequence-level metric, and it is not what the model was trained to optimize. :::
Both halves of that follow from the same fact. Perplexity is a monotone function of average negative log-likelihood,
$$
  \operatorname{PPL} = \exp\left(-\frac{1}{N}\sum_t \log p(y_t \mid y_{<t})\right),
$$
so a model encouraged to avoid extreme probabilities will assign somewhat less probability to the correct token even when it knows the answer, and that likelihood-based metric duly gets worse. But translation quality is not the same thing as confidence on individual training tokens. A less overconfident model can generalize better and produce better sequence-level output.

Label smoothing also has an interesting future. Müller, Kornblith, and Hinton :::cite Müller, Kornblith, and Hinton. **When Does Label Smoothing Help?** 2019. [arXiv](https://arxiv.org/abs/1906.02629) ::: later found that although it can improve generalization and calibration, it can make a trained network a *worse teacher* for knowledge distillation. Their representation analysis suggests that smoothing erases some of the information carried by the relative probabilities among incorrect classes.

### A complete training step
We can finally assemble the procedure. Take a minibatch of source-target sentence pairs, and then:
1. **Tokenize.** Raw strings become discrete sequences $x_1,\ldots,x_n$ and $y_1,\ldots,y_m$.
2. **Embed.** Each token retrieves a learned vector $e_t = E_{x_t}$, scaled by $\sqrt{d_{\text{model}}}$ and combined with positional information.
3. **Encode the source.** The encoder repeatedly applies self-attention and feed-forward transformations, producing contextual source representations.
4. **Shift the target right.** The decoder receives $\langle\mathrm{BOS}\rangle, y_1, \ldots, y_{m-1}$, and causal masking ensures each position may use only its correct prefix.
5. **Run the decoder.** Masked self-attention communicates among earlier target positions; cross-attention retrieves from the encoded source; the feed-forward layers transform what results.
6. **Project to logits.** $\ell_t = Eh_t$, under weight tying.
7. **Normalize into probabilities.** $p_t = \operatorname{softmax}(\ell_t)$.
8. **Compare against the correct next tokens.** With label smoothing, $\mathcal{L} = -\sum_t\sum_v q'_{t,v}\log p_{t,v}$.
9. **Backpropagate.** Compute $\nabla_\theta \mathcal{L}$ for every trainable parameter: $E, W^Q, W^K, W^V, W^O, W_1, W_2, \gamma, \beta$, and the rest.
10. **Update with Adam,** using the scheduled learning rate.

Then do it again, a hundred thousand times. Twelve hours later you have a translator, and not one line of the procedure above mentioned translation.

### What has actually been learned?
Almost nothing we discussed in the architecture section is directly supervised. Nobody labels antecedent heads, syntax heads, useful embedding dimensions, or feed-forward features. There is no $\mathcal{L}_{\text{syntax}}$, no $\mathcal{L}_{\text{facts}}$, no $\mathcal{L}_{\text{reasoning}}$. The only externally supplied signal is, in effect, *given this context, you should have assigned more probability to this token*. Every piece of internal structure exists because it happened to reduce that quantity.

Which does, in fairness, explain a great deal. To predict
> The capital of France is ___
it is useful to encode facts about France. To predict
> Because the trophy was too large for the suitcase, it did not ___
it is useful to represent which object could fit inside which. To predict source code, it is useful to infer syntax and program structure. To predict an argument, it may be useful to represent relationships that look suspiciously like reasoning.

That is simultaneously the extraordinary power and the extraordinary limitation of the objective: the model is rewarded for whatever internal machinery helps prediction, and not at all for discovering machinery we would find intelligible. Whether these representations deserve the names we give them, and how far next-token prediction can push such abilities, are entirely separate questions—ones we will have to return to when we reach interpretability.

At every position we ask the same question, **what comes next?**, measure the error, differentiate it, and adjust several million numbers. Then we ask again.

The next step in the genealogy was to ask what happens if we stop treating it as a translation system at all: throw away the encoder, discard cross-attention, point the decoder at an enormous pile of unlabelled text, and make next-token prediction itself the entire pretraining task.
> **what if prediction is the only task?**

That brings us to **GPT-1**.
:::progress 2026-08-13T11:07:20.998Z