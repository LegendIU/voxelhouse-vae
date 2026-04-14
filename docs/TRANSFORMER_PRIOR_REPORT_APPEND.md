## Engineering and MLOps Layer

To improve reproducibility and experimental rigor, we added a lightweight MLOps layer to the
latent-token Transformer prior pipeline. All train/eval/sample stages are now configuration-driven,
write explicit manifests, and export structured metrics in both CSV and JSONL formats.

In addition, checkpoint management was standardized via named aliases (`best_by_val_loss`,
`best_by_val_ppl`, `last`), which makes reported comparisons easier to audit and reproduce.
A small smoke-test CI layer was also added to verify core latent prior functionality such as
conditioning, decoding strategies, and tensor-shape correctness.

These additions do not change the underlying generative objective, but they significantly improve
experimental hygiene, reproducibility, and the reliability of comparisons between Gaussian and
learned priors.
