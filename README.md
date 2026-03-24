# Causal-HalBench: Uncovering LVLMs Object Hallucinations Through Causal Intervention(AAAI 2026)

<img src="./assets/intro.jpg" alt="image-20230517233229650" style="zoom:80%;" />

This repository is official implementation for our paper: [Causal-HalBench: Uncovering LVLMs Object Hallucinations Through Causal Intervention(AAAI 2026)](https://arxiv.org/abs/2511.10268)

<br>

> **Abstract:** *Large Vision-Language Models (LVLMs) often suffer from object hallucination, making erroneous judgments about the presence of objects in images. We propose this primarily stems from spurious correlations arising when models strongly associate highly co-occurring objects during training, leading to hallucinated objects influenced by visual context. Current benchmarks mainly focus on hallucination detection but lack a formal characterization and quantitative evaluation of spurious correlations in LVLMs. To address this, we introduce causal analysis into the object recognition scenario of LVLMs, establishing a Structural Causal Model (SCM). Utilizing the language of causality, we formally define spurious correlations arising from co-occurrence bias. To quantify the influence induced by these spurious correlations, we develop Causal-HalBench, a benchmark specifically constructed with counterfactual samples and integrated with comprehensive causal metrics designed to assess model robustness against spurious correlations. Concurrently, we propose an extensible pipeline for the construction of these counterfactual samples, leveraging the capabilities of proprietary LVLMs and Text-to-Image (T2I) models for their generation. Our evaluations on mainstream LVLMs using CausalHalBench demonstrate these models exhibit susceptibility to spurious correlations, albeit to varying extents.*


## Evaluation 
### 1. Causal-HalBench dataset download
