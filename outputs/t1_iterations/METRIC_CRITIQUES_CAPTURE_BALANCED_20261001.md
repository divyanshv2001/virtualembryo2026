# Capture-balanced batch: four post-experiment agent critiques

Completed: 2026-10-01T09:31:57.152356+00:00
Private result SHA256: 13410a7c94083490c5c04dfdee22c461644b09b708de0c5b3d439fe4f4b1e6f7

No experiment rerun or extra reward. These are four actual fresh-context critic-agent reviews of the existing completed batch.

## de_score

Neither new method merits promotion. Cellweighted DE skills.4668/.4304 and capturemedian.4628/.4381 are below copy.5 on both folds; raw DE negative. Reused folds do not establish independent embryo transfer. Detection calibration may absorb capture artifacts. Refit all base components excluding heldcapture and report paired DE/direction by capture, not cell independence.

## de_direction

Cellweighted raw direction.0013/-.1402, capturemedian-.0110/-.0936, below anchor.0195/.1068. Capture balancing did not recover direction. Detection calibration may improve marginals while worsening signs. Freeze temporal tests, refit exposed base components, compare copy/anchor and report capture/lineage direction uncertainty.

## mmd_u

Cellweighted MMD worse than copy on both folds; capturemedian worse first, barely better second (.27364/.23642 versus copy.26304/.23757). Anchor remains stronger (.25065/.20635). Detection calibration may fit capture sparsity rather than biology. Refit excluding heldcapture, use permitted current observations only for calibration, compare paired temporal MMD with capture-level uncertainty.

## variogram

Cellweighted errors.010032/.010119 and capturemedian.010275/.009420 exceed copy.009574/.008673 on both folds; skills below.5. Detection-odds changes may alter sparsity/apparent structure without biology. Refit base excluding heldcapture, calibrate from permitted current anchor only, freeze before future scoring; require lower errors versus controls on both folds.

## Design adjudication

Some critic wording excluded the heldcapture even from current-anchor calibration. For observed-anchor deployment, exclude heldcapture from fitted encoder/dynamics/response learners while allowing its CURRENT observed expression for predeclared calibration. Future expression never fits or selects. Jev selected this boundary (confidence1.0),785input/62output tokens,1781-byte packet. This does not certify embryo independence or a score gain.

No automatic retries or call-count inflation. More Jev calls are allowed for distinct meaningful design, selection, diagnosis and synthesis decisions; repeated packets are cached.
