# Paper-alignment update (2026-11-12)

Reference: RL_CIM_XIAONAN_information_sciences (8).pdf and the supplied update ZIP.

Implemented: replacement main entry point and main parameter table; disjoint target/competing seeds; candidate-count cap and lower bound; competitor-priority synchronous irreversible diffusion in HG_Diffusion_SICP and HG_Diffusion_SICP_pos; local per-simulation RNG; explicit next-action replay masks and zero terminal bootstrap; input order [xi,phi]; exploration reaches .01; disjoint reward/evaluation seed ranges; explicit --budget option. Removed obsolete CompeteMain.py. Updated basic S2V training constants from300/.01/.8/35 to600/.03/.9/50.

Budget default remains the supplied historical 1% convention, not a uniquely specified budget for every figure. Use --budget for the intended figure. The paper describes multiple target budgets. Old legacy tracking functions and specialized HCL/beta/plot workflows have not all been audited for complete behavioral equivalence. Full-space S2V's learner/masking requires a separate audit. This is not a claim that all legacy scripts now reproduce all manuscript results.

Validation: three tiny tests for competitor tie precedence, synchronous propagation, terminal bootstrap, and empty legal-mask rejection; main CLI and Python syntax checks. No full-network training or evaluation rerun. Old numerical results, runtime tables and changed repetition counts are not validated by this code update. The PDF still contains empirical claims that require original evidence or corrected reruns. Original caches are read unchanged; this does not certify their distance correctness.
