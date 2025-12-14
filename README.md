# AG News Headline Classification: From Keywords to DistilBERT

This repository contains a mini end-to-end AI pipeline for classifying news headlines into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.
The goal is to compare a very simple **keyword-based baseline** with a fine-tuned **DistilBERT** model on the same dataset and metrics.

Repository contents:

* `AI Model Pipeline.ipynb` – main Jupyter notebook (data loading, baseline, DistilBERT pipeline, evaluation).
* `AI Model Pipeline_Report.pdf` – LaTeX report compiled from the homework template.
* `README.md` – this project summary.

---

## 1. Task description and motivation

**Task.**
Given a short English news headline, predict which of the four AG News topics it belongs to: World, Sports, Business, or Sci/Tech. This is a standard single-sentence, single-label text classification problem.

**Motivation.**
News topic classification is a classic example where both humans and machines need to understand *what the article is really about* rather than just spotting a few obvious words. It is simple enough to run on a single GPU, but still large enough to show a clear gap between hand-written rules and modern pretrained language models. This makes it a good playground for a “mini AI pipeline” assignment.

**Input / output.**

* **Input:** one news headline (short English text string).
* **Output:** one of four labels: `World`, `Sports`, `Business`, or `Sci/Tech`.

**Success criteria.**

* Quantitatively, we mainly look at **accuracy** and **macro-F1** on the test set.
* Qualitatively, we want the model to make reasonable decisions on tricky headlines (mixed topics, financial news with political context, tech news with company names, etc.), and to clearly outperform the naïve keyword baseline.

---

## 2. Dataset description

**Source.**
We use the AG News corpus as provided by the Hugging Face datasets library ("ag_news"). Each example contains a short news text with one of four topic labels; in this project we use the headline/text field as the model input.

**Size and splits.**

* Official training split: **120,000** labeled headlines.
* Official test split: **7,600** labeled headlines.
* From the 120k training examples, we create a **90/10 split**:

  * **Training:** 108,000 examples
  * **Validation:** 12,000 examples
  * **Test:** 7,600 examples (official test split, untouched)

**Preprocessing.**

* **For the keyword baseline**

  * Lowercase headline
  * Remove basic punctuation
  * Split on whitespace into tokens
  * Count matches against hand-crafted keyword lists for each class

* **For the DistilBERT pipeline**

  * Use `distilbert-base-uncased` tokenizer (`AutoTokenizer` from Hugging Face)
  * Lowercasing and subword tokenization handled by the tokenizer
  * Truncate/pad to a maximum length of **64 tokens**
  * Produce `input_ids` and `attention_mask` tensors
  * Labels are already integers `{0,1,2,3}`, mapped to the four classes

---

## 3. Baseline and AI pipeline design

### 3.1 Naïve keyword baseline

**Idea.**
For each topic (World, Sports, Business, Sci/Tech) we manually create a short list of **keywords** that are typical of that domain (e.g., “football”, “Olympics” for Sports; “stocks”, “oil”, “market” for Business, etc).

**Prediction rule.**

1. Preprocess the headline (lowercase, strip punctuation, split into tokens).
2. For each class, count how many of its keywords appear in the headline.
3. Predict the class with the **highest keyword count**.
4. If no keywords are found for any class, we default to World (label 0) as a simple fallback.

**Why this is naïve.**

* No parameters are learned from data; everything is hard-coded.
* The model ignores word order, syntax, and longer-range context.
* It cannot recognize synonyms or paraphrases outside the keyword lists.
* It tends to over-predict `World` for headlines that mention countries, cities or politicians, even when the real topic is business or technology.

This baseline is extremely easy to implement and serves as a sanity check, but clearly under-uses the information in the dataset.

---

### 3.2 DistilBERT-based AI pipeline

We build a small fine-tuning pipeline on top of Hugging Face Transformers.

**Models used.**

* `distilbert-base-uncased` (Hugging Face)
* `AutoTokenizer` for tokenization
* `AutoModelForSequenceClassification` with 4 output labels (World / Sports / Business / Sci-Tech)

**Pipeline stages.**

1. **Input preprocessing** – pass raw headline strings to the tokenizer with `padding="max_length"`, `truncation=True`, `max_length=64`.
2. **Encoding** – DistilBERT produces contextual embeddings for each token and a pooled representation for the sentence.
3. **Classification** – a randomly initialized linear layer maps the pooled representation to 4 logits; `argmax` gives the predicted label.
4. **Training & evaluation** – fine-tune the whole model using cross-entropy loss on the training set, monitor validation metrics, and finally evaluate on the held-out test set.

**Hyperparameters.**

* Batch size: **32** (train and evaluation)
* Epochs: **5**
* Learning rate: **5e-5**
* Weight decay: **0.01**
* Warmup ratio: **0.05**
* Optimizer & training loop handled by `transformers.Trainer` / `TrainingArguments`

This setup is small enough to run on a single GPU (e.g., RTX-3090) in reasonable time but still powerful enough to show a big improvement over the keyword rules.

**Environment (rough).**

* `Python 3.10.12`
* `PyTorch 2.3.1+cu12`
* `transformers`
* `datasets`
* `scikit-learn`
* `accelerate`
* `numpy`

All imports and installation hints are included at the top of the notebook.

---

## 4. Metrics, results, and comparison

### 4.1 Metrics

Because this is a 4-way classification problem, we use:

* **Accuracy** – fraction of correctly classified headlines.
* **Macro-F1** – F1-score averaged over the four classes, giving equal weight to each topic even if class frequencies differ.

These two metrics together give a good picture of overall performance and whether any class is being neglected.

### 4.2 Quantitative results

On the **AG News test set**, we obtain:

| Method              | Accuracy | Macro-F1 |
| ------------------- | -------: | -------: |
| Keyword baseline    |   0.5332 |   0.5263 |
| DistilBERT pipeline |   0.9413 |   0.9413 |

So the DistilBERT pipeline improves accuracy and macro-F1 by roughly **+0.41** and **+0.42** respectively compared to the keyword baseline.

The notebook also prints the full classification report for DistilBERT (per-class precision/recall/F1); those numbers are consistent with the summary above.

### 4.3 Qualitative examples

We also looked at individual headlines where the two methods behave differently. A few representative cases:

1. **Indian PM pledges to protect poor from oil-driven inflation**

   * Gold: *Business*
   * Baseline: *World* (focuses on country / politician)
   * DistilBERT: *Business* (picks up economic cues like “oil-driven inflation”).

2. **TV ads for a desktop Linux distribution**

   * Gold: *Sci/Tech*
   * Baseline: *World* (product name not in any keyword list)
   * DistilBERT: *Sci/Tech* (understands the tech context).

3. **Selling Houston “warts and all”, urban afflictions described**

   * Gold: *Business*
   * Baseline: *World*
   * DistilBERT: *Business*, focusing on image/marketing rather than generic world news.

4. **South Korea and Singapore seal a free-trade pact**

   * Gold: *Business*
   * Baseline: *Business*
   * DistilBERT: *World* – shows that even the neural model can be biased by strong location words and underweight the economic phrase “free-trade pact”.

5. **U.S. oil reserves and rising oil prices**

   * Gold: *Business*
   * Baseline: *World*
   * DistilBERT: *Business*, using words like “oil prices” and “barrel”.

Overall, these examples match the quantitative picture: the baseline catches some obvious sports and world-news patterns, but the DistilBERT model is much better at subtle financial and technology topics.

---

## 5. Reflection and limitations

* The DistilBERT pipeline performed **better than expected**: with only five epochs and standard hyperparameters, it already reached around 0.94 accuracy and macro-F1.
* Designing the keyword baseline was more work than it first looked; choosing keyword lists, resolving overlaps between categories, and handling headlines with no matched keywords all required manual trial-and-error.
* The most annoying technical part of the pipeline was getting the dataset splits, tokenization, and `Trainer` configuration to line up so that training and evaluation ran smoothly on the GPU.
* Accuracy and macro-F1 were useful summary metrics: when they improved, the qualitative examples almost always looked better too, and macro-F1 helped ensure that no single class was being ignored.
* However, these metrics do not distinguish between “harmless” confusions (e.g., Sports vs. World in a borderline sports-politics story) and more serious topic mistakes (e.g., Sci/Tech vs. Business), so they miss some nuance.
* With more time or compute, natural extensions would be:

  * trying a slightly larger model such as BERT-base,
  * doing light domain adaptation for business/technology news, and
  * adding a stronger classical baseline (e.g., TF-IDF + logistic regression) between the keyword rules and DistilBERT to make the performance gap more interpretable.
