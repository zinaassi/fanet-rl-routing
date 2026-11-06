Phase 2c round 1 checkpoints (scripts/train_c_ppo_round1.py)
=============================================================

Each run keeps its latest two checkpoints here:

    out/checkpoints/2c_round1/<run>/ckpt_<update>.pt

<run> is one of: control, spacing_med, spacing_small, spacing_large.
A checkpoint is written every 25 updates and holds EVERYTHING needed to go on:
actor, critic, both Adam optimisers, the value normaliser, the numpy and
torch random generator states, the update number, the best-so-far validation
score and its update, the training log and the validation curve so far.


TO MOVE A RUN TO ANOTHER MACHINE
--------------------------------
1. On the new machine, check out the same branch and commit (phase-2c), so
   the code is identical. Results are only bit-for-bit reproducible on the
   same code; different hardware or library versions may change the last
   digits.

2. Copy, keeping the same paths under rl_sim/:

     out/checkpoints/2c_round1/<run>/ckpt_*.pt      (the latest one is enough)
     out/models/2c/c_actor_r1_<run>.pt              (best-so-far actor)
     out/models/2c/c_critic_r1_<run>.pt             (its critic)
     out/2c_r1_curve_<run>.csv, out/2c_r1_log_<run>.csv   (optional; they
                                                    are rewritten from the
                                                    checkpoint)

   The best-so-far model files MUST travel with the checkpoint: the
   checkpoint remembers the best score, and the model saved at that score is
   only in out/models/2c/. If they are missing, the run continues but the
   final "re-validate the best actor" check will stop with an error.

   The start models (c_actor_curriculum_direct.pt, c_critic_curriculum_
   direct.pt) are needed only to START a run, not to resume one. Every
   finished run also needs out/models/2c/c_actor_curriculum_direct.pt for
   the edge-case step (the edge-case script loads it as its default).

3. From rl_sim/, resume:

     python -u scripts/train_c_ppo_round1.py --runs <run> --resume --workers <N>

   --workers only sets the pool size; it does not change the results.
   Several runs can be listed after --runs; each resumes from its own latest
   checkpoint or, if it has none, starts fresh from the direct model.

4. After a run reaches update 400 the script re-validates its best actor,
   runs the edge cases on it and rewrites the summary tables. To rebuild the
   tables alone:

     python scripts/train_c_ppo_round1.py --report-only
