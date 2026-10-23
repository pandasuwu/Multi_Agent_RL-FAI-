Repo: this folder (Multi_Agent_RL-FAI-), CS329 course project. Two learning agents play the war of attrition. Their policy networks are wired like the fruit fly give-up circuit (MaleCNS: PPL1 -> MBON18/19 -> AVLP727m, 258 neurons, 4,952 edges). We compare the real wiring against a degree-preserving rewire and a random sparse graph. Game A: V = c = 1, NE Exp(1). Game B: private values v ~ Exp(1), BNE v^2/2. The metric is exact exploitability on a 64-point grid.

Before writing anything, read README.md, RESULTS.md, docs/theory.md, configs/base.yaml, configs/*.yaml, src/agents/policies.py, src/agents/factory.py, src/training/trainer.py, src/training/runner.py, src/metrics/stats.py and scripts/run_experiment.py. Follow the existing conventions: yaml configs with lr_from, tuned_lr.yaml, results/<experiment>/runs, the stats in src/metrics/stats.py, and the parameter-match assert.

Ground rules
- Work on a branch called flygm-ppo. Commit after every milestone and run `git push -u origin flygm-ppo` after every commit. An earlier version of this work was lost because it was never pushed.
- Do not touch existing results or configs. Every new experiment gets a new config and a new results directory.
- Add tests for every new piece, and keep `pytest` green.

1. FlyGM arm (frozen synapses, after Jin et al. 2026, arXiv:2602.17997)
- W[post, pre] = sign(pre) * synapse_count, scaled so the spectral radius is 1. W is a fixed buffer and is never trained.
- What trains: a per-neuron descriptor eta_i in R^9, one shared update MLP f_psi(h_i, m_i, eta_i) with m = W h, the input encoder and gate on the PPL1 neurons, and the readout from AVLP727m to 64 logits. Unroll 6 steps, the same as the other circuit arms.
- Size f_psi so that the trainable parameter count is within 5% of the connectome arm (5,610). Assert this the same way the other arms do.
- Build it for the real, rewired and random graphs. Rewired and random graphs are redrawn per seed, as in the existing arms.
- Sign of unsigned neurons: all 16 PPL1 neurons are dopaminergic and have node_sign 0 in data/connectome/give_up_circuit.npz (26 unsigned neurons in total). With sign 0, W has no PPL1 output at all. Make this a config option, unsigned_sign in {0, +1}, default +1, and record which value each run used.
- Diagnostic: for each graph, log the input-to-output gain sum over k = 1..6 of |(W^k)[AVLP727m, PPL1]|. A quick check gave about 0.008 for the real graph and about 4-13 for rewired and random graphs (unsigned_sign = +1). Also log the spectral radius before scaling (about 1,230 for the real graph), because the global scaling may be what crushes the PPL1 path.

2. PPO learner for sampled mode
- Add it as a training option next to REINFORCE: 4 clipped updates per round on the same batch, clip epsilon 0.2, no entropy bonus. Use the existing running-mean baseline as the critic.
- It must work for every network arm, including FlyGM. Keep the loss_win_ratio weighting so it still works under asymmetry.

3. Game A exact rerun at full time resolution
- Rerun the existing game_a_exact grid (all 9 arms, 20 seeds, the same tuned learning rates) with eval_every: 1, into a new results directory.
- Goal: settle whether the connectome's 19-24 episode speed-up to threshold 0.01 (RESULTS.md section 4) is real. Under eval_every: 25 every crossing time is a multiple of 25.

4. Experiments
- Tune learning rates for FlyGM (all 3 graphs) and for the PPO variants with scripts/tune_lr.py and the existing protocol (tuning seeds 1000+). In sampled mode, extend the network grid down to 0.0001, because dense sat on the 0.0003 edge.
- FlyGM real / rewired / random in all four settings (A exact, A sampled, B exact, B sampled): 20 seeds in exact mode, 10 in sampled mode, the same step counts as the existing configs.
- PPO for every network arm in A sampled and B sampled, 10 seeds.
- Shared learning-rate robustness runs for the circuit arms, FlyGM included, like configs/robust_*_shared_lr.yaml.
- If time allows: PPO under loss_win_ratio in {2, 4, 8} in Game A sampled.
- Record the wall time of every experiment.

5. Analysis and write-up
- Use the existing analysis and stats code: per-arm tables, pairwise contrasts with bootstrap CI, Hedges' g and Holm p.
- Add one figure for FlyGM vs the trained-edge circuits and one for REINFORCE vs PPO.
- Add RESULTS.md section 9 (FlyGM), section 10 (PPO) and section 11 (the Game A eval_every: 1 result). Each section states its compute setup and caveats.
- Update the README arm table. Commit and push.

Stop and ask me before changing anything in the game definitions, the circuit extraction, or the existing arms.
