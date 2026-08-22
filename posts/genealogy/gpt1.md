---
title: The genealogy of language models
date: 2026-08-21
description: What if prediction is the only task?
timezone: Asia/Kolkata
series: genealogy
series_order: 3
permalink: genealogy/gpt1
---

## What if prediction is the only task?
Up until now our discussion has been informed by the original Transformer paper and the task it set out to solve: translation. Nonetheless, the mechanics of attention and learning are not specific to translation specific; the only thing that made it adept at french or german is that every sentence fed was paired with a english one (roughly four and a half million of them for English-German, and thirty-six million for English-French). Data like that, *labelled*, back then, was tediously assembled by hand and there is no way around it.

Radford et al. :::cite Radford, Narasimhan, Salimans, and Sutskever. **Improving Language Understanding by Generative Pre-Training.** 2018. [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) ::: motivate GPT-1 by the fact that unlabelled text is abundant while labelled text is scarce, and that we ought therefore to arrange for the majority of the learning to happen on the former. It takes two stages:
1. A Transformer decoder is trained to predict the next token over a large corpus of text.
2. For each downstream task, a single linear layer is attached to the pre-trained model and the whole thing is fine-tuned on some labelled data.

With a vocabulary of $40{,}478$ tokens, an unlearned model pays a negative log-likelihood of $\ln 40478 \approx 10.6$ nats on every prediction. The pre-trained model reports a token-level perplexity of $18.4$, which corresponds to a loss of $\ln 18.4 \approx 2.9$. :::note This is entirely off the unlabeled data. ::: The resulting model improved on the state of the art in nine of the twelve benchmarks the authors evaluated, using one architecture across all of them and essentially no task-specific components.

### What was 2018 like?
Before we talk about GPT-1, I believe it is worth catching up on the revelations of the two years of the original Transformer architecture and this paper.

The idea of pre-training then fine-tuning was not itself novel. Dai and Le :::cite Dai and Le. **Semi-supervised Sequence Learning.** 2015. [arXiv](https://arxiv.org/abs/1511.01432) || Howard and Ruder. **Universal Language Model Fine-tuning for Text Classification.** 2018. [arXiv](https://arxiv.org/abs/1801.06146) || Peters et al. **Deep Contextualized Word Representations.** 2018. [arXiv](https://arxiv.org/abs/1802.05365) ::: had essentially proposed this in 2015, and Howard and Ruder's ULMFiT used fine-tuning for text classification, though both were built on LSTMs. Peters et al. trained bidirectional LSTM language models and then used their internal states as *features*, which were fed into whatever task-specific architecture the practitioner had already constructed. The representations transferred; the model did not.

The other concept generally attributed is making Transformers decoder-only. For posterity I will note here that Liu et al. :::cite Liu et al. **Generating Wikipedia by Summarizing Long Sequences.** 2018. [arXiv](https://arxiv.org/abs/1801.10198) ::: had already discarded the encoder for long-form summarization.

To me there seems to be parallel between the perception of GPT-1 & Transformers and what they actually did. Both are touted as these landmark papers in the space (I agree!) but what is revolutionary about them is the construction of the models not the ideas. Both of them took concepts that were common-place and the forefront of research, dropped some previously highly-regarded obvious (recurrence and convolution for Transformers; labelled data and encoders for GPT-1), and showed incredible promise.

### Goodbye Encoder!
Recall the three applications of multi-head attention in the original architecture:
1. encoder self-attention
2. masked decoder self-attention
3. encoder-decoder cross-attention
As I alluded to in the first section, we would revisit some of these mechanisms. GPT-1 retains only the masked decoder—getting rid of the encoder entirely. Cross-attention exists so that the decoder may consult a source sentence, and encoder self-attention exists so that the source sentence has a representation worth consulting. If the task is simply to continue a passage of text, there is no source sentence, and both operations are left without anything to do.

What remains is a stack of blocks containing two sub-layers apiece: :::note GPT-1 remains post-norm. :::
$$
  A^{(l)} = \operatorname{LayerNorm}\!\left(H^{(l-1)} + \operatorname{MaskedMultiHead}(H^{(l-1)}, H^{(l-1)}, H^{(l-1)})\right)
$$
 
$$
  H^{(l)} = \operatorname{LayerNorm}\!\left(A^{(l)} + \operatorname{FFN}(A^{(l)})\right).
$$

The paper compresses the entire forward pass into three lines:
$$
  h_0 = UW_e + W_p
$$
 
$$
  h_l = \operatorname{transformer\_block}(h_{l-1}), \quad l \in [1, n]
$$
 
$$
  P(u) = \operatorname{softmax}(h_n W_e^\top)
$$
where $W_e$ is the token embedding matrix and $W_p$ the position embedding matrix. It is worth observing that the output projection in the final line is $W_e^\top$, the transpose of the input embedding matrix. This is exactly weight tying.

#### Counting parameters
The model is composed of twelve layers, $d_{\text{model}} = 768$, twelve attention heads, $d_{\text{ff}} = 3072$, and a context length of $512$.

Within each layer :::note This ignores biases together with the LayerNorm gain and shift parameters, which amount collectively to a few tens of thousands. :::, attention contributes four square projections, $W^Q$, $W^K$, $W^V$, and $W^O$, giving $4 \times 768^2 = 2{,}359{,}296$, while the feed-forward network contributes two rectangular ones, $2 \times 768 \times 3072 = 4{,}718{,}592$, for a total of $7{,}077{,}888$ parameters per block, or $84{,}934{,}656$ across the twelve. The paper specifies a byte-pair encoding with $40{,}000$ merges; once base characters and special tokens are included the released vocabulary contains $40{,}478$ entries, so the embedding matrix accounts for $40{,}478 \times 768 = 31{,}087{,}104$, and the learned positional table for a further $512 \times 768 = 393{,}216$. Summing these gives $116{,}414{,}976$.

Something worth noticing in our decomposition is that roughly twenty-seven percent of the model consists of the embedding matrix, which is to say that better than a quarter of GPT-1's capacity is a *lookup table*. This ratio inverts as models grow, and it is one of the more reliable ways to see at a glance how large a model is.
 
Training proceeded for 100 epochs on minibatches of 64 contiguous sequences of 512 tokens, which is $32{,}768$ tokens per optimization step.
:::progress 2026-08-22T13:15:24.978Z