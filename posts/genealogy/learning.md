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

### Weight Tying
In what we have described above, it seems to follow that we would have one embedding matrix
$$
  E \in \mathbb{R}^{V \times d_{\text{model}}}
$$
that maps tokens to vectors, and an "output" matrix
$$
  W_{\text{out}} \in \mathbb{R}^{V \times d_{\text{model}}}
$$
that maps hidden vectors against possible output tokens.

However, I ask you to reframe this. The row $E_v$ asks, in effect
> what vector should represent token $v$ when I read it?
while $(W_{\text{out}})_v$ asks
> what direction in representation space should make me believe the answer is token $v$?
Press and Wolf :::cite [Press and Wolf](https://arxiv.org/pdf/1608.05859) ::: showed that the output matrix of a neural language model itself behaves as a meaningful word embedding, and found that tying the input and output matrices could improve language-model perplexity while substantially reducing parameter count. In neural translation models, their experiments found that weight tying could reduce model size dramatically without degrading performance. Inan, Khosravi, and Socher independently arrived at closely related weight-sharing results from a loss-based analysis. :::cite [Inan, Khosravi, and Socher](https://arxiv.org/pdf/1611.01462) :::

We therefore impose
$$
  W_{\text{out}} = E.
$$
With our convention that vocabulary embeddings occupy the rows of $E$, logits become
$$
  \ell_t = Eh_t.
$$

The original Transformer take weight tying one step further: it shares the weight matrix between both embedding layers and the pre-softmax output transformation; this commonly called three-way weight tying. 

Weight tying reduces capacity in exchange for parameter efficiency and an inductive bias that says these two problems ought to share structure. With that being said, there is no mathematical necessity that the best space for interpreting an input token must be identical to the best space for classifying an output token.

We can now read symbols, transform them, and convert the final representations into probabilities.

We still need to define what counts as being wrong.

### The objective
We are still basking in the sun of 2017, and so the ensuing discussion about training is not what we do with modern day large language models but instead focused on translation.

Let $x = \langle x_1, \ldots, x_n \rangle$ be the source sentence and $y = \langle y_1, \ldots, y_m \rangle$ be the target translation. Our goal is to for the model to assign a high probability to the entire correct translation: $p_\theta(y \mid x)$.

Using one of our earlier remarks, the probability assigned to an entire translation is equivalently the product of the probabilities assigned to each correct next token:
$$
  p(y \mid x) = p(y_1 \mid x) \cdot p(y_2 \mid y_1, x) \cdot p(y_3 \mid y_2, y_1, x) \cdot \dots.
$$
The product of many small probabilities is numerically awkward and requires a high level of floating point precision, so we log both sides:
$$
  \log p_\theta(y \mid x) = \prod_{t=0}^{m-1} \log p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right).
$$
Training for the maximum likelihood estimation (choosing parameters that maximize the quantity above over the training set), we posit
$$
  \theta^\ast = \arg\max_{\theta} \sum_{(x,y) \in \mathcal{D}}\sum_{t=1}^{|y|} \log p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right).
$$
Optimization libraries conventionally minimize rather than maximize, so we negate it:
$$
  \mathcal{L}_{\text{NLL}} = -\sum_{(x,y) \in \mathcal{D}}\sum_{t=1}^{|y|} \log p_\theta\!\left(y_{t+1} \mid y_{\leq t}, x\right).
$$
This is the **negative log-likelihood**. For a one-hot target distribution, this is also exactly the usual cross-entropy loss.

It is worth noting for our discussions down-the-line, if we remove the encoder and the source sentence $x$, the objective becomes
$$
  p_\theta(x_1,\ldots,x_n) = \prod_{t=1}^n p_\theta(x_t \mid x_{< t}),
$$
with loss
$$
  \mathcal{L} = - \sum_t \log p_\theta\!\left(x_{t} \mid x_{< t}\right).
$$
That is, essentially, the objective that will eventually train many LLMs.

#### Cross-entropy
Suppose the correct next token is $y$. Let $q$ be its one-hot target distribution:
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
because only one entry of $q$ is nonzero.

Suppose, the model assigns the correct token probability as $p_y = 0.9$, then the cross-entropy loss is approximately $0.105$. Alternatively, if the model assigns $p_y = 0.01$, the loss would be approximately $4.605$. Thus, severely wrong predictions are appropriately measured.

There is a nice property to make note off.

Let $\ell_j$ denote the logit for vocabulary item $j$. Combining softmax and cross-entropy gives
$$
  \frac{\partial \mathcal{L}}{\partial \ell_j} = p_j - q_j.
$$
For the correct token, in our case $y$, the loss becomes $p_y - 1$ which is negative unless the model already assigns probability $1$ to the answer. Therefore, gradient descent will *push upwards*.

For every incorrect token, $p_j - q_j > 0$, so so gradient descent *pushes downward*. The more probability the model incorrectly assigns to some alternative, the larger the correction.

#### Causal shifting
Suppose our desired target sequence is $y_1, y_2, y_3, y_4$. To train
$$
  p(y_3 \mid y_2,y_1,x),
$$
the model must be shown $y_1$ and $y_2$ but not $y_3$. To train
$$
  p(y_4 \mid y_3,y_2,y_1,x),
$$
it must see $y_3$.

One naive solution would be to run the model once to predict $y_1$, again to predict $y_2$, again to predict $y_3$, and so forth. That would throw away precisely the parallelism for which we constructed the Transformer.

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

The causal attention mask we introduced earlier prevents position $t$ from looking to its right. Therefore:
1. position $1$ sees `<BOS>` and predicts $y_1$;
2. position $2$ sees `<BOS>`, $y_1$ and predicts $y_2$;
3. position $3$ sees `<BOS>`, $y_1,$ $y_2$ and predicts $y_3$;

Generation will be sequential; training does not have to be.

#### Teacher forcing
There is a vital choice made in our discussion so far, that is worth explicitly pointing out now: teacher forcing. That is, using the *true* value rather than the network's own previous output, and feeding it back into the subsequent computation during training. Spelling it out with an example, suppose the target begins
> The transformer unpacks...
When learning to predict *unpacks*, we condition the model on the correct prefix
> The transformer
regardless of whether the model itself would actually have predicted *transformer* one step earlier.

Teacher forcing combines particularly well with causal masking: since every correct previous token is known, the entire shifted target can enter the decoder at once.

A sentence containing $m$ target tokens therefore gives us roughly $m$ supervised next-token prediction problems in one forward pass. This is one of the reasons the deceptively simple next-token objective scales so effectively.
:::progress 2026-08-11T11:45:36.271Z

### Training & Inference
What happens when we are no longer training? More precisely, during training, when predicting token $t$, irrespective of what the model picked earlier, it receives $y_{< t}$, the correct prefix. But during inference :::definition **Inference** is the process of using an already trained large language model to generate text or answers for new user prompts. ::: the model receives $\hat y_{< t}$, its own generated prefix, one that could be incorrect.

This train-test mismatch became known as **exposure bias**. Bengio et al. :::cite [Bengio et al.](https://arxiv.org/pdf/1506.03099) ::: proposed scheduled sampling: gradually replace some ground-truth previous tokens during training with tokens generated by the model itself, thereby exposing the model to the kinds of prefixes it will encounter at inference.

Huszár subsequently showed that the scheduled-sampling objective is not, in general, a consistent estimator of the data distribution: the apparent fix can alter what distribution the model is incentivized to learn. :::cite [Huszár](https://arxiv.org/pdf/1511.05101) :::
:::remark A subtle but important point
It is common to say "teacher forcing causes exposure bias, therefore maximum likelihood is flawed."

The autoregressive factorization is exact. If every learned conditional matched the true conditional distribution, sampling autoregressively from the model would reproduce the correct joint distribution.

The practical problem is that our learned conditionals are imperfect. During rollout, an imperfect prediction may place us in regions of sequence space where those approximations are worse, allowing errors to compound.

So there is a genuine train-generation mismatch, but simply replacing maximum likelihood with model-generated prefixes does not automatically solve it.
:::
This issue will return much later under other names and other training paradigms.

#### How does loss reach attention heads?
Going back to our example sentence from the previous section, "Because the trophy was too large for the suitcase, it did not fit.", suppose one of our attention heads eventually learns that *it* should retrieve information from *trophy*. How does it learn about this mistake? **Backpropagation**.

Backpropagation conveys the fact that some later layer realizes that the model assigned an insufficient probability mass to the correct translated token, by carrying the errors backwards through the computation graph.

Consider a single attention operation written as $Z = AV$, where
$$
  A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$
Suppose the gradient arriving from later layers is
$$
  G = \frac{\partial \mathcal{L}}{\partial Z},
$$
then using the definition of $Z$, the gradient with respect to the values $V$ is
$$
  \frac{\partial \mathcal L}{\partial V} = A^\top G.
$$
The forward attention weights therefore also influence where learning signal is routed backward through the value pathway. Similarly,
$$
  \frac{\partial \mathcal L}{\partial A} = GV^\top.
$$
That error propagates backward through the softmax and then through $QK^\top$, producing gradients for both $Q$ and $K$ and ultimately for $W^Q$, $W^K$, and $W^V$. Since those matrices were themselves produced from previous hidden representations, the gradient continues backward through every preceding Transformer block, all the way to the embeddings.

The process therefore looks something like
1. At each target position, the model produces a probability distribution over the vocabulary. The loss compares this distribution with the correct next token and measures how much probability the model failed to assign to the desired answer.
2. For softmax with cross-entropy, this begins with the signal $\frac{\partial \mathcal L}{\partial \ell_j}=p_j-q_j$. The correct token receives a negative gradient when its probability is too low, while incorrect tokens receive positive gradients proportional to how much probability the model assigned them.
3. Since the logits are produced from the final decoder representation, $\ell_t=W_{\text{out}}h_t$, the logit gradients determine how the final hidden state $h_t$ should have changed to make the correct token more probable. If the output weights are tied to the embedding matrix, this step also contributes directly to the gradient of that shared matrix.
4. The gradient flows backward through the feed-forward network, residual connections, layer normalization, and cross-attention and masked self-attention sublayers. Each operation receives a signal describing how changing its output would have changed the final loss.
5. For an attention operation, gradients flow both through the retrieved information $V$ and through the attention weights $A$. Thus the model can learn both what information should be carried and where it should have retrieved that information from.
6. The gradients through $A$ propagate through the softmax and the score matrix $QK^\top$, producing gradients for $Q$ and $K$. Together with the gradient through $V$, these ultimately produce gradients for $W^Q$, $W^K$, and $W^V$.
7. Because $Q$, $K$, and $V$ were constructed from hidden representations produced by previous layers, the gradient does not stop at a single attention head. It propagates recursively through earlier attention blocks and feed-forward networks, assigning credit or blame to every computation that contributed to the prediction.
8. Eventually the gradient arrives at the representations with which the computation began. The embedding vectors for tokens that appeared in the example are adjusted according to how changes in those vectors would have affected the final prediction. Over many examples, tokens whose representations are useful in similar contexts can consequently acquire related structure.
9. After backpropagation, the model has a gradient $\nabla_\theta\mathcal L$ describing, locally, how the loss would change if each parameter were perturbed. This includes the embedding matrix, attention projections, feed-forward weights, normalization parameters, and output projection.
10. Backpropagation tells us the direction in which each parameter affects the loss; it does not by itself decide how large an update to make. An optimizer takes these gradients and converts them into parameter updates. In the original Transformer, that optimizer was Adam, together with a carefully chosen learning-rate schedule.

Given all of these gradients, how should we actually update the model?

#### How does it improve?
Training uses minibatches. Sequences in a batch generally need compatible tensor shapes, so shorter sequences are padded.

The paper describes batch size in tokens, not merely number of sentences. A batch of 100 ten-token sequences has very different computational cost from a batch of 100 thousand-token sequences.

For each minibatch we obtain a scalar loss $\mathcal L(\theta)$, differentiate it
$$
  g_t = \nabla_theta \mathcal L(\theta_t),
$$
and then need to decide how far to move the parameters. Naively,
$$
  \theta_{t+1} = \theta_t - \eta g_t,
$$
which is **gradient descent**. The transfomer uses something far more intricate: **Adam**. :::cite Kingma and Ba. Adam [A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980) :::

Suppose one parameter repeatedly receives gradients like $\{0.0001, 0.0002, 0.0002, \ldots \}$ while another receives $\{8, -1, 4, \ldots \}$. Using exactly the same effective update scale for both can make optimization difficult. Adam (adaptive moment estimation) maintains running statistics for every parameter.

Given gradient, $g_t$, Adam first tracks an exponential moving average of the gradient:
$$
  m_t = \beta_1m_{t-1} + (1-\beta_1)g_t.
$$
This is an estimate of its first moment. Informally, $m_t$ behaves somewhat like momentum: it smooths the noisy minibatch gradient and preserves information about its recent direction. Adam also maintains an exponential moving average of the squared gradient:
$$
  v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2.
$$
This estimates the second raw moment. Parameters whose gradients have historically been large therefore accumulate larger $v_t$.

Because both moving averages begin at zero, early estimates are biased toward zero. Adam corrects this:
$$
  \hat m_t = \frac{m_t}{1-\beta_1^t} \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}.
$$
The final update is
$$
  \theta_t = \theta_{t-1} - \eta_t \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
$$
The denominator is what makes Adam adaptive.
:::progress 2026-08-12T06:15:31.114Z