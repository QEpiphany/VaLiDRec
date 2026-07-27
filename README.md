# VaLiDRec: Variable-length LLM-aligned Semantic IDs for Generative Recommendation

## Abstract

Generative recommendation commonly represents items with semantic identifiers (SIDs). Existing methods typically construct fixed-length SIDs through clustering and quantization, producing artificial latent codes that compress item semantics, are misaligned with pretrained large language model (LLM) vocabularies, and require costly autoregressive decoding during inference. These limitations hinder semantic expressiveness, cold-start recommendation, and retrieval efficiency.

In light of this, we propose **VaLiDRec**, a generative recommendation framework based on variable-length, LLM-aligned semantic identifiers. VaLiDRec constructs SIDs directly from informative native LLM vocabulary tokens via token importance estimation, semantic-quality-aware pruning, and collision-aware refinement, allowing identifier lengths to adapt to item semantic complexity. To model user preferences, VaLiDRec incorporates graph-aware soft prompts and reformulates recommendation as token-set prediction with inverted-index retrieval, eliminating autoregressive SID generation and beam search.

Experiments on four real-world datasets show that VaLiDRec consistently outperforms strong sequential and generative recommendation baselines across all evaluation metrics. It further achieves superior zero-shot item cold-start performance and **87.07× faster inference** than LC-Rec. These results demonstrate that LLM-native variable-length semantic identifiers provide a more expressive and efficient paradigm for generative recommendation.

## Framework

<p align="center">
  <img src="Figures/Framework.png" width="95%" alt="Framework of VaLiDRec">
</p>

<p align="center">
  <em>
    Overview of VaLiDRec. Stage 1 constructs variable-length semantic identifiers directly from native LLM vocabulary tokens through token importance estimation, semantic-quality-aware pruning, and collision-aware refinement. Stage 2 incorporates graph-aware soft prompts and formulates recommendation as token-set prediction followed by efficient inverted-index retrieval.
  </em>
</p>
