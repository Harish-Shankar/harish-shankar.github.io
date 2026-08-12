---
title: Markdown element sample
date: 2026-08-06
description: A reference page for every element supported by the custom Markdown parser.
status: reference
timezone: Asia/Kolkata
---

# Markdown element sample

This page contains a paragraph with **strong text**, *emphasized text*, ~~struck text~~, `inline code`, [an inline link](https://example.com), and inline math $p_\theta(x_t \mid x_{<t})$. :::note Margin notes are anchored to the exact phrase they clarify. :::

## A second-level heading

### A third-level heading

#### A fourth-level heading

##### A fifth-level heading

###### A sixth-level heading

> A blockquote for quotations, excerpts, or a thought that deserves a distinct rhythm.

:::definition Attention head
An **attention head** maps a query and a set of key-value pairs to a weighted sum of the values:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$
:::

:::remark Why keep a remark collapsible?
Remarks carry useful context that is not required for the main argument. Collapsing them preserves the article's reading rhythm while leaving the detail close at hand.
:::

- An unordered list item
- A second item with **emphasis**
- [x] A completed task item
- [ ] An incomplete task item

1. A numbered step
2. Another numbered step
3. A final numbered step

```python
def attention(query, key, value):
    # A deliberately small example.
    scores = query @ key.T
    return softmax(scores) @ value
```

Display mathematics occupies its own line:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

| Element | Example use |
| --- | --- |
| Table row | Comparing related ideas |
| Another row | Organizing compact data |

![Image placeholder](placeholder)

The placeholder can carry a source citation :::cite Vaswani et al. **Attention Is All You Need.** 2017. [arXiv](https://arxiv.org/abs/1706.03762) ::: without interrupting the article.

---

Progress markers can separate writing sessions without implying that the post is finished. :::progress 2026-08-08T06:24:05.000Z :::
