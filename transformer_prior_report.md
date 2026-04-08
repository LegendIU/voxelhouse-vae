# Transformer Prior Report Notes

## What Changed

- Added a 3D `VQ-VAE` that converts voxel grids into discrete latent tokens.
- Added an autoregressive Transformer prior over those tokens.
- Added unconditional and optionally conditioned sampling with `greedy`, `temperature`, `top-k`, and `top-p`.
- Added benchmark scripts that compare the original Gaussian VAE prior against the learned Transformer prior.

## Why Learned Prior Beats Isotropic Gaussian

The original VAE baseline samples `z ~ N(0, I)`, which assumes latent dimensions are independent and globally Gaussian. That is convenient for optimization, but it is a weak generative assumption for structured 3D houses.

The learned Transformer prior is stronger because it models:

- token-level dependencies across the latent grid instead of independent dimensions
- multimodal structure without forcing everything into a single isotropic Gaussian
- conditional factorization `p(z) = Π p(z_t | z_<t, c)` with optional house attributes / class / style tokens

In practice, this gives a better inductive bias for:

- coherent global structure
- sharper mode coverage
- controllable sampling via decoding strategy and condition tokens

## What To Report

For the final write-up, the strongest comparison is:

1. `Gaussian prior from VAE`
2. `Learned Transformer prior from VQ-VAE tokens`

Recommended evidence:

- token-level validation loss / perplexity for the prior
- qualitative sample grids for both models
- decoding sweep: `greedy`, `temperature`, `top-k`, `top-p`
- diversity metrics: `unique_ratio`, pairwise Hamming, pairwise IoU diversity
- plausibility metrics: validity, connected components, largest connected-component ratio
- optional nearest-reference IoU against the held-out split

## Suggested Narrative

The VAE baseline is reconstruction-first and only weakly generative because its latent prior is fixed. The VQ-VAE + Transformer stack becomes generative in the full autoregressive sense: it learns a discrete latent vocabulary and then learns the distribution over that vocabulary directly.
