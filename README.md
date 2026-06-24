[Day2_README (1).md](https://github.com/user-attachments/files/29298307/Day2_README.1.md)
# Fashion-Image-search-optimization# Gen AI Bootcamp — Day 2 Homework
## AI Product Intelligence System — Recommendations, Deduplication & Text Search

**Author:** Sonu
**Kaggle Notebook:** https://www.kaggle.com/code/sonikasree/notebook8686d3fd2b

---

## What's in this submission

| File | Description |
|---|---|
| `notebook8686d3fd2b.ipynb` | Full source code — all 3 tasks, run on Kaggle with GPU |
| `Day2_Homework_Report.docx` | Short report explaining the approach + screenshots for each task |

---

## Overview

This project extends a Day 1 multimodal product-understanding system into a full **AI Product Intelligence System**, using **CLIP (ViT-B/32)** embeddings as a shared foundation for three advanced features:

1. **Smart Product Recommendation Engine** — suggests complementary products (not just visually similar ones)
2. **Unique Product Catalog Creation** — detects and removes near-duplicate product listings
3. **Reverse Product Search** — search the catalog using a text query instead of an image

**Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (Myntra dataset, ~44,000 products) — 800 products sampled for this submission.

---

## Task 1: Smart Product Recommendation Engine

**Approach:** Instead of recommending visually similar items, the system matches products sharing the same `usage` context (e.g. Sports, Ethnic, Casual) and `gender`, but from a **different category** — capturing items that are genuinely bought together.

**Example:**
- Input: *Mother Earth Women Classic Maroon Kurta* (Ethnic, Women's Apparel)
- Recommended: *Royal Diadem Gold & Purple Earrings*, *Fabindia Black & Gold Conical Purse*

---

## Task 2: Unique Product Catalog Creation

**Approach:** CLIP image embeddings are compared with cosine similarity, then grouped using **DBSCAN clustering** (`eps = 0.05`) to detect near-identical product photos. One representative product is kept per cluster.

**Results (on 800-product sample):**

| Metric | Value |
|---|---|
| Original product count | 800 |
| Unique catalog count | 667 |
| Duplicates removed | 133 (16.6%) |

---

## Task 3: Reverse Product Search

**Approach:** CLIP places images and text in the same embedding space, so a search query is embedded as text and compared directly against product image embeddings using cosine similarity — no separate search engine needed.

**Example queries tested:** `"blue casual shirt"`, `"red kurta"`, `"men's leather shoes"`, `"gold jewellery"` — all returned relevant, correctly ranked results.

---

## Key technical note

`transformers` version `5.0.0` changed the behavior of `CLIPModel.get_image_features()` — it now returns a `BaseModelOutputWithPooling` object instead of a plain tensor. This was fixed by calling `model.vision_model()` and `model.visual_projection()` directly (and the text-side equivalents for search) to get proper 512-dimensional embeddings in CLIP's shared space.

---

## How to run

1. Open the notebook on Kaggle (link above) or upload the `.ipynb` to your own Kaggle account
2. Attach the dataset: **Fashion Product Images (Small)** by `paramaggarwal`
3. Enable GPU (Settings → Accelerator → GPU T4 x2)
4. Run all cells top to bottom
