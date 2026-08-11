---
title: The genealogy of language models
date: 2026-08-11
description: How do models actually learn?
timezone: Asia/Kolkata
series: genealogy
series_order: 2
permalink: genealogy/learning
---

## Learning
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