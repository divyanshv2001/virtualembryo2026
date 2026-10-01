# Mean/mass projection feasibility: four specialist reviews

Observation-only audit mean_mass_feasibility_01; no temporal forecasts, benchmark metrics or reward. All four cases failed the fixed200-iteration convergence gate. Maximum gene log-mean residual4.30–4.75, raw row-mass relative error approximately1e-15. This establishes failure of this numerical procedure, not mathematical infeasibility. Locked support and original margins were never relaxed or called successful.

DE specialist: Problem—one margin passes while the other fails. Proposed solution—check necessary support/mass bounds, then a jointly scaled constrained solver with analytic derivatives/multiple starts if bounds permit. Validation—both tolerances, support and nonnegativity simultaneously; gene means alone do not preserve Mann-Whitney ranks. No DE score available.

Direction specialist: Problem—raw mass conservation cannot establish projection validity or temporal direction. Proposed solution—damped exact-constraint solver and continuation from a verified feasible initialization, past/current only. Validation—matched support, input, tolerances/budget with known-feasible and unchanged controls; any residual breach stays invalid. No direction metric available.

MMD specialist: Problem—numerical nonconvergence gives no distribution benefit. Proposed solution—scale/damp a joint solver, compare row-only and mean-only diagnostics using past data. Validation—both margins in every case, frozen budget/controls; even convergence cannot establish MMD gain. No MMD metric available.

Nonspatial CSS specialist: Problem—matching margins would not certify cell-state contrast preservation. Proposed solution—joint constraints, analytic derivatives, scaled residuals and explicit termination diagnostics. Validation—both tolerances plus a separately predeclared contrast bound; no future tuning. No CSS metric available.

Fresh DE/direction agents reviewed summaries. A fresh MMD spawn returned the observed error `agent thread limit reached`; existing completed MMD/CSS specialists were reused with only the compact new packet and instructions not to load prior history. No scientific resource or quota-reset time was inferred from this limit.

Jev failure diagnosis selected necessary_bounds_first,confidence1.0,1570-byte request,actual703input/46outputtokens. For n rows and a gene with target log-sum L and m locked-positive cells, convexity requires at least m*expm1(L/m) total raw expression. Sum these minima cannot exceed available raw row-mass total. Each gene's required log-sum also cannot exceed sum(log1p(row_mass)) over supported rows. A violated necessary bound rejects the exact locked constraints; passing leaves feasibility unresolved. Reference, constructed Jensen-impossible, empty-support and unresolved controls passed tests. Register the current-only four-case bounds audit before any solver or temporal test. No score/reward change and no claim of measured credit savings.
