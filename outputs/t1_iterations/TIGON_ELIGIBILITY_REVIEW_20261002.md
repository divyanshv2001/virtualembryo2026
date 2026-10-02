# TIGON eligibility entry review

The [TIGON paper](https://www.nature.com/articles/s42256-023-00763-w) models velocity and growth jointly using Wasserstein–Fisher–Rao transport. Its density inputs require population masses; absent other information, the authors use collected cell counts. The paper links [author code](https://github.com/yutongo/TIGON). The repository reports PyTorch/neural ODE implementation and recommends GPU acceleration.

Our metadata-only screen uses<=8.0 rows:1963/3000/3000 selected cells across7.5/7.75/8.0, with6/4/4assay IDs. Counts are capped/domain-selected; assay IDs are not independent embryo IDs. Absolute growth is not identifiable from these numbers. Provided-label composition changes are descriptive and can reflect sampling or annotation. No expression or future composition entered this screen.

A conditional equal-total-mass density/reweighting adaptation remains a research hypothesis, not faithful biological-growth inference. Pin/review numerical and objective code, check whether normalization supports this adaptation, then measure bounded synthetic gradients/runtime/memory with growth-disabled control. No author data/weights, real training or score promotion yet. See NEXT_TIGON_PROTOCOL_PREFLIGHT.json. Original scorer and readiness gates unchanged.
