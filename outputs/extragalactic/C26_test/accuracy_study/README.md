# What limits the accuracy of the stream posteriors (study of 2026-11-24)

Question: the networks trained on `train_v2` recover few parameters well and their posteriors reweight poorly
on the observed tracks. Is that the summary network, the flow, the training set, or the simulator?

Everything here refers to `train_v2` (10,000,000 streams, 30,000 particles, 99 leapfrog steps) and, unless a
run is named, to the network `npe_trainv2_1000epoch_vr4_5e4lr` (spline flow 8 x 512, transformer 64 x 2) at
its saved epoch 381, held-out loss 11.36 nats.

## Short answer

1. **Short tracks (up to about 12 bins) are already right.** There the simulation set is dense enough to give
   the posterior without a network (weighted simulations), and the network's marginals agree with it: posterior
   widths within 2 to 4 %, means within 0.1 sd, for all 13 physical parameters. The parameters the network does
   not recover (R_s, halo orientation and q, r_s) are not constrained by such a track. No architecture changes
   that.
2. **Long tracks are limited by the training set, not by the network.** For a track of more than 25 bins only
   a handful of the 10 million simulations fit at the level of the errors (effective sample 1 to 9 for every
   observed track that long), the network fits tracks cut from its own training streams better than held-out
   ones (0.5 to 0.8 nats; 1.3 to 1.6 for the MLP summary), and with a quarter of the streams the loss on long
   tracks is 1 nat worse.
3. **A quarter of the training examples come from streams the simulator does not resolve** (energy error of the
   progenitor orbit above 1 % with 99 leapfrog steps; 36 % of the long-track examples). Their track jumps when
   the parameters move by 1e-4 of the prior width, so the exact likelihood is rough there and neither
   importance sampling nor nested sampling can be trusted for them. Removing them does not make the network
   better on the rest; simulating them with more steps would.
4. **The far ends of long streams are not too noisy.** Particle noise is 0.03 of the data error per bin near the
   progenitor and 0.08 at 30 to 48 bins from it; the second guard keeps it there. No trimming is needed. The
   guard is what costs: it removes 5 times (short tracks) to 16 times (long tracks) the simulations that could
   otherwise be compared with an observed track. Simulating good prior samples again with 3 and 10 times the
   particles gives back 1.6 to 6 times the usable samples for long tracks and better fits among them, without
   changing what the network infers from a given window (section 7).
5. **Two exact symmetries are unused.** The simulator is scale-free: one simulation stands for a one-parameter
   (dissolved progenitor) or two-parameter (surviving progenitor) family of parameter sets with the same track
   up to a factor. And the model has a handedness (the progenitor moves anticlockwise) that the observed tracks
   need not share: for 9 of the 32 tracks the simulations fit better with the angle reversed.
6. **Within the network, the summary transformer is the part worth enlarging** (twice as wide and deep: 0.35
   nats better, 0.7 on long tracks). A larger coupling flow gains nothing; a diffusion model in its place gains
   0.25 nats. None of these changes which parameters are recovered.

## What was measured

| question | finding | figure |
|---|---|---|
| Which examples does the network learn from? | 50 % resolved thin streams (loss 9.7 nats), 12 % wide clouds (15.2), 12 % folded (12.2), 26 % unresolved (12.4) | `01_loss_by_class.png` |
| Are long streams noisy? | particle noise no; unresolved orbits yes | `02_track_noise.png` |
| How far is the network from the exact posterior? | importance-sampling efficiency 12 % (short), 5 % (medium), 1 % (long) on mock tracks, the same for three networks | `03_importance_sampling.png` |
| Is the training set large enough? | not for long tracks | `04_train_vs_heldout.png`, `07b_data_size.png` |
| Do the simulations look like the observed tracks? | short tracks yes; most long ones have no simulation within their errors | `05_gallery_p1.png`, `05_gallery_p2.png`, `05_prior_weights_summary.png` |
| Unused symmetries? | mass, length and time scalings; the sense of the angle | `06_symmetries.png`, `05_prior_weights_summary.png` |
| Summary network or flow? Diffusion? | see "Training experiments" | `07_experiments.png` |
| Do more particles make the long streams better? | yes for what can be used of them and how well they fit; no for what the network infers from a given window | `10_particle_number.png`, `11_particle_gallery.png`, `09_good_samples.png` |
| Can particles be stored cheaply for augmentation at training time? | 19 kB per stream with an ordered truncated SVD (360 kB raw), track error about the particle noise; too slow to rebuild at every step | `08_particle_svd.png` |

## 1. Long streams: what is noisy and what is not

`scripts/extragalactic/track_noise.py`: 6,000 mock tracks of held-out streams, each stream simulated again with
another random key for the particle release, and with the same key and the parameters shifted by 1e-6, 1e-4
and 1e-2 standard deviations of the prior. Numbers in `numbers/track_noise.json`.

- **Another key** moves the model track by 0.13 of the data error per bin (median over mocks; 90th percentile
  0.32) and the log-likelihood of the mock by 0.2 to 0.5 (half the 16-84 range: 0.17 for short tracks, 0.45
  for long ones). It is the same for every kind of stream. Per bin it grows slowly with the distance from the
  progenitor: median 0.03 of the data error in the first three bins, 0.08 at 30 to 48 bins, 99th percentile
  about 1 everywhere. The realised noise is 1.25 times what the second guard assumes (it takes hw / sqrt(n),
  the standard error of a mean; a median has 1.25 times that), uniformly, also in bins that hold two pieces
  of a folded stream.
- A new key makes the true parameters fail a guard (likelihood 0) in 4 % of short, 8 % of medium and 13 % of
  long mock tracks.
- **A shift of the parameters** by 1e-4 prior sd leaves the track of a resolved stream where it was (shift
  0.00 of the data error; 0.8 % fail a guard). For an unresolved stream 15 % then fail a guard and another
  9 % change their log-likelihood by more than 2. At 1e-2 prior sd: 34 % fail and 29 % move by more than 2
  (resolved thin streams: 3 % and 2 %, which is the smooth dependence on the parameters).
- The jumps start at an energy error of 1 % (0.1 to 1 %: 1.5 % fail at a shift of 1e-4; 1 to 10 %: 10 %;
  above 10 %: 27 %), so the 1 % of `report.md` is the right line. 26 % of the training examples are above it:
  18 % of the short ones, 28 % of the medium, 36 % of the long.
- For the observed tracks, 28 % (tracks up to 16 bins) to 42 % (longer) of the network's posterior draws are
  unresolved streams; for IC1657, NGC4256, NGC5263, PGC000902 and UGC01424 it is 75 to 92 %.

What this means: the unresolved streams are an artefact of the integrator (report.md: their tracks change by
a median of 30 % when the steps are raised) and their likelihood is a rough function of the parameters. The
network learns its average, which is the sensible answer, but no sampler of the exact likelihood can confirm
it. Mock tracks whose true stream is unresolved reweight with an efficiency of 1.0 %, resolved ones with 8.5 %.

## 2. How good is the network where it can be checked

`scripts/extragalactic/sbi/prior_weights.py` weights every usable simulation of the set by its likelihood for a
track (progenitor position and sigma_sys integrated out). That is the posterior with no network in it, good
where many simulations carry weight. Numbers in `numbers/prior_weights.json`.

- 26 mock tracks and 4 observed tracks have an effective sample of 100 or more simulations. On them the
  network's posterior sd over the weighted-simulation sd is 0.98 to 1.04 for each of the 13 parameters
  (median over tracks), and the means differ by 0.02 to 0.13 sd. The evidence from the network's importance
  weights agrees with the evidence from the set for the 19 observed tracks where both have an effective sample
  above 3: median difference 0.14 nats, 16 within 1, the largest 1.7 (NGC1162).
- The importance-sampling efficiency of the network's posterior on 32 mock tracks is 12 % (6-12 bins), 5 %
  (13-25) and 1.3 % (longer), and the same within noise for the transformer at 94 and 381 epochs and for the
  MLP summary. From the weights, the network's posterior is about 1.2 nats (short), 2.0 (medium) and more than
  3.3 (long) away from the exact one in KL divergence. So the one-dimensional marginals are right and the
  joint is not yet: the loss could still fall by that much.

- **NGC1162 against nested sampling** (28 bins; the dynesty run finished on 2026-11-24: log Z -80.69 +- 0.10,
  8.4 million likelihood calls, 9.6 hours on four GPUs). The network's marginals are close to it: the medians of
  all 15 parameters lie within 0.56 nested sd, and the widths within 20 % except for v_y0 (network 1.6 times
  wider), log M (1.3) and t_int (1.2). The evidence from the network's importance weights is -80.3 (20,000
  draws, effective sample 4.5) and -81.2 (1,000 draws); from the simulation set -83.0 (effective sample 6,
  biased low). So also on a long observed track the marginals are usable, while the joint is far from exact
  (importance-sampling efficiency 0.02 %).

## 3. Is the training set large enough

- **Effective sample of the set at the observed tracks.** Median 5 simulations; 4 tracks of 32 have 100 or
  more, 10 have 10 or more, and every track of 26 bins or more has fewer than 10 (most 1 to 5), although 5,000
  to 60,000 simulations pass the guards for each. For mock tracks: median 4,000 (short), 57 (medium), 19
  (long). The observed long tracks are harder than mock ones of the same length: in the gallery the best
  simulations miss several of them by a few sigma along whole sections (NGC0804, NGC1162, NGC4256, NGC5263).
- **Where the set has no simulations the network has no posterior**: the effective sample of 1,000 network
  draws follows that of the 7 million simulations track by track (right panel of `03_importance_sampling.png`).
  For four long tracks the network's draws miss the best simulations of the set by 35 to 175 in
  log-likelihood (PGC1092512, PGC938075, PGC000902, UGC01424), and for NGC5263 by 13.
- **Train against held-out** (`numbers/train_vs_heldout.txt`): the loss on mock tracks cut from training
  streams is lower than on held-out streams by 0.01 nats (6-12 bins), 0.17 (13-25), 0.52 (26-35) and 0.65
  (36-49); for the MLP summary 0.04, 0.27, 0.83, 1.29. Tracks of 36 bins or more can be cut from 0.6 % of the
  streams only.

## 4. Resolution of the simulations

- **Steps.** 99 leapfrog steps for the whole of t_int leave 26 % of the examples unresolved (section 1). This
  is the resolution problem that matters.
- **Particles.** The second guard (model variance below a quarter of the data variance) is where the particle
  number enters. With the guard off, the simulations that can be compared with an observed track are 4.8
  times as many for tracks up to 16 bins, 12.9 times for 17 to 28 bins and 16.3 times for longer ones, and
  the evidence of the long tracks rises by 11 nats (median): the excluded simulations include better fits
  than any allowed one. Three times the particles (guard at 4/3) gives 1.6, 2.0 and 2.2 times as many
  simulations and +0.4, +2.4 and +6.3 in log-evidence. The effective sample of the long tracks stays at 1 to 3
  either way.
- **Bins.** A track in bins shifted by half a bin cannot be had from the stored track by averaging neighbours
  (median error 1.2 times the particle noise, but 15 times at the 90th percentile and 11 % of the bins lost).
  The mock observations always put the progenitor on a bin edge; the observed tracks do not.

## 5. Symmetries

`scripts/extragalactic/symmetries.py` simulates 3,000 streams again with scaled parameters and the same key.

| scaling | parameters | track | surviving progenitor | dissolved progenitor |
|---|---|---|---|---|
| mass | masses x U, velocities x sqrt U, t_int / sqrt U | unchanged | exact (99 % of resolved streams identical to 1e-7) | broken: 3 % median, 18 % at the 90th percentile |
| length | lengths x L, velocities / sqrt L, t_int x L^1.5 | x L | exact | exact |
| both, t_int fixed | lengths and velocities x L, masses x L^3 | x L | exact | broken as above |

- The mass scaling is broken for a dissolved progenitor only because its mass is ramped to a fixed 1 Msun; a
  ramp to a fixed fraction of the initial mass would make it exact too.
- Unresolved streams follow the scalings in 75 to 80 % of the cases; the rest jump, as under any change.
- **Consequences.** (a) For a surviving progenitor the track holds no information at all along
  (log M, log m, log v, log t) = (1, 1, 1/2, -1/2): the posterior there is the prior, whatever the network.
  (b) Every simulation can be reused at other scales: a draw (L, U) from the prior along its orbit gives a new
  exact training example, the track multiplied by L, at no simulation cost. The ranges are set by the prior
  box: R_s 10 to 25 kpc allows a factor 2.5 in L in all, t_int 1 to 4 Gyr a factor 16 in U. (c) The same holds
  for the nested sampler: two of its dimensions (one for dissolved progenitors) can be integrated without a
  simulation.
- **Sense of the angle.** The prior has v_y0 > 0: the progenitor moves anticlockwise and the leading arm is at
  positive angle. A stream that moves the other way on the sky is the mirror image, which the prior excludes.
  With the angle of each observed track reversed, the evidence from the set is higher by more than 2 for 9
  tracks (PGC000902 +27, PGC1092512 +18, NGC5263 +14, UGC01424 +8.5, IC1657 +4.9, NGC4632 +4.4, UGC09239 +3.4,
  ESO356-012 +2.4, PGC039258 +2.1), lower by more than 2 for 8, and within 2 for 15. These are from effective
  samples of 1 to 5, so the sizes are rough; that the sense matters is not.
- Rotations about the line of sight and the mirror along it are already taken out by the frame (data.py).

## 6. Storing particles for augmentation at training time

`scripts/extragalactic/particle_svd.py` (400 streams of `train_v2` with stored particles) and
`particle_store.py` (a new pilot set, `data/particles_pilot/`: 5,000 prior draws, seed 2602, all particles).

- **Naive truncated SVD fails.** The matrix release epoch x particle has independent scatter in every entry:
  rank 16 (51 kB) is off by 5 times the particle noise in the median bin, and rank 64 keeps nothing.
- **Ordered SVD works.** Sort the particles by angle, cut them into chunks of 50 neighbours, keep the angle of
  every 20th particle, the mean radius of every chunk, and the rank-4 SVD of the sorted residuals of the
  chunks: 18.8 kB per stream as float32 against 360 kB of particles (240 kB for angle and radius), 188 GB for
  10 million streams. The track made from it differs from the true one by 0.7 times the particle noise in
  the median bin (2.9 at the 90th percentile; 4 % of the bins, thin ones, fall below 10 particles), and
  equally for bins shifted by half a bin or half as wide. Rank 16 (50 kB) gives 0.46. A random subset of the
  particles at the same size loses half the bins.
- Half precision needs a scale per stream: stored as plain float16 the radii lose their digits and some
  streams overflow.
- **Speed.** Rebuilding the particles and binning them in JAX runs at 9,400 streams per second on the H200
  (shared with three trainings): 0.44 s for a batch of 4096, against 0.02 s for a training step. So not at
  every step as it stands. It would serve to refresh the stored tracks between epochs, or with fewer rebuilt
  particles.

## 7. More particles: a test on good prior samples

`scripts/extragalactic/particle_number.py`. Good prior samples were taken from `train_v2` itself: 5,000 streams
with a resolved orbit and a run of at least 26 bins with enough particles (such streams are 644,256 of the
7.08 million usable ones, 9.1 %), and the 2,883 simulations that fit the 16 observed tracks of more than 16
bins best when the particle-noise guard is off. These 7,871 parameter sets were simulated again with 30,000
(control), 90,000 and, for 1,801 of them, 300,000 particles. Guard 1 is kept at the same fraction of the
particles. Numbers in `numbers/particle_number.json`; figures `09_good_samples.png`, `10_particle_number.png`,
`11_particle_gallery.png`.

| particles | long streams that keep 26 usable bins at a data error of 1 kpc (0.3 / 3 kpc) | samples that count for a long observed track (median of 16 tracks) | the same for the two longest tracks (46, 48 bins) | best log-likelihood against 30,000 (median; largest) | network loss on the same long mock tracks |
|---|---|---|---|---|---|
| 30,000 | 30 % (5 % / 66 %) | 1 | 1 | 0 | 8.08 nats |
| 90,000 | 52 % (14 % / 79 %) | x 1.6 | x 2.6 to 3.1 | +2.0; +27 | +0.003 +- 0.007 |
| 300,000 | 68 % (29 % / 88 %) | x 2.0 | x 5.2 to 6.1 | +3.6; +68 | +0.010 +- 0.016 |

- **More particles lengthen what can be used of a long stream.** The mean usable run at a data error of 1 kpc
  goes from 20 bins to 26 and 31. This is the second guard letting go as the particle noise falls.
- **More of the good samples count for the long observed tracks, and better ones.** Among the same parameter
  sets the best log-likelihood rises by 2.0 (median) with 90,000 particles and 3.6 with 300,000; for
  UGC01424, PGC1001085, PGC938075 and PGC000902 by 12 to 32, and for NGC4256 by 68 (its best sample at 30,000
  particles is a poor one). In the gallery the best samples at 300,000 particles reach the peaks of UGC08717,
  PGC1001085 and PGC938075 that those allowed at 30,000 miss. The best fits with more particles come up to the
  best fits at 30,000 with the second guard off (equal for UGC01424, PGC938075, PGC000902 and IC0120, 1 to 4
  short for the others) and not beyond: nothing new is found, what the guard hid becomes usable.
- **What the network infers from a given window does not change.** With the model track of 90,000 or 300,000
  particles under the same window, errors and noise, the loss on mock tracks of 26 bins or more is the same
  within 0.01 nats. The particle noise of 30,000 particles is not what limits the inference.
- **Cost.** A simulation costs in proportion to its particles. Three times the particles gives 1.6 times the
  usable samples for a typical long track and about 3 times for the longest; ten times gives 2 and 5 to 6. So
  90,000 particles pay for themselves on the longest tracks and 300,000 do not. They matter beyond the count,
  though: for the long tracks the evidence sits mostly in simulations the guard excludes at 30,000 particles
  (section 4: +6.3 in log-evidence with the guard three times looser), and more simulations at 30,000
  particles would not reach those.
- **Where the good samples are** (`09_good_samples.png`): lower halo mass (median log M 12.26 against 12.61,
  almost none above 13.3), heavier progenitor (log m 8.41 against 8.12), closer in (x0 42 kpc against 76),
  lower v_y0 (142 against 189 km/s), longer t_int. Halo shape and orientation are as in the prior.
- **A bug found on the way.** `generate.py` stored the particle count of a bin as a 16-bit integer, which wraps
  above 32,767: 37 % of the streams at 90,000 particles and 82 % at 300,000 had negative counts. Fixed (32-bit
  above 32,767 particles; `mock.py` and `prior_weights.py` no longer assume 16 bits). The numbers above are
  from simulations made after the fix. `train_v2` (30,000 particles) is not affected.

## 8. Training experiments

All on the first quarter of `train_v2` (500 shards, 1.59 M usable training streams), 100 epochs = 38,900 steps
of batch 4096 with the cosine schedule, compared on the same 20,000 mock tracks of held-out streams
(`scripts/extragalactic/sbi/nll.py`; the loss of a diffusion or flow-matching model is its density from the
probability-flow ODE). Numbers in `numbers/study.json`.

Difference of the held-out loss from the baseline's, in nats, paired over the mocks (negative = better); the
standard errors are 0.01 to 0.05. "Resolved" = mocks whose stream has a resolved orbit.

| run | change from the baseline | parameters | all | resolved, 6-12 bins | resolved, 13-25 | resolved, 26-49 | unresolved |
|---|---|---:|---:|---:|---:|---:|---:|
| `q_base` | spline flow 8 x 512, transformer 64 x 2: loss 12.19 | 8.1 M | 0 | 0 | 0 | 0 | 0 |
| `q_sum128` | transformer 128 x 4 | 8.8 M | -0.35 | -0.13 | -0.44 | -0.69 | -0.35 |
| `q_dm` | diffusion (MLP 6 x 512) in place of the flow | 5.6 M | -0.25 | -0.11 | -0.31 | -0.36 | -0.32 |
| `q_fm` | flow matching (MLP 6 x 512); 4,096 mocks | 5.6 M | -0.19 | -0.04 | -0.22 | -0.33 | -0.28 |
| `q_flowbig` | spline flow 12 x 1024 | 36.6 M | +0.10 | +0.14 | +0.06 | +0.12 | +0.06 |
| `q_resolved` | trained on resolved orbits only (`de_max: 0.01`) | 8.1 M | +1.06 | -0.02 | +0.17 | +1.08 | +3.18 |
| `s_data06` | 1/16 of the set instead of 1/4, same number of steps | 8.1 M | +1.59 | +0.96 | +1.59 | +3.07 | +1.52 |

- **The summary transformer is the part of the network that limits it.** Twice the width and twice the blocks
  gain 0.35 nats, 0.69 on long tracks, for 0.7 M more parameters (and about 2.3 times the time of a step).
- **A larger coupling flow gains nothing**: 4.5 times the parameters, 0.10 nats worse.
- **Diffusion and flow matching do better than the coupling flow with the same summary** (0.25 and 0.19 nats),
  so the coupling flow is a limit of its own that size does not lift. The price: their density is an ODE solve,
  22 minutes for 20,000 densities against seconds, which is what importance weights need. A larger transformer
  with a diffusion model was not run.
- **None of this changes which parameters are recovered.** The correlation of the posterior mean with the truth
  moves by at most 0.04 between the runs (log M 0.43 to 0.46, R_s 0.08 to 0.11, x0 0.92 to 0.93), and the
  posterior widths by a few per cent (middle and right panels of `07_experiments.png`). The gains are in the
  joint shape of the posterior.
- **Training on resolved orbits only does not help on resolved orbits**: equal on short tracks, 1.1 nats worse on
  long ones. It has 37 % fewer streams and overfits (training loss 11.0, held-out 12.0), which is the
  training-set limit again.
- **A sixteenth of the set is far worse than a quarter** (1.6 nats, 3.1 on long tracks) and overfits badly:
  training loss 9.4 against 15.3 held out at the end, best held-out epoch 13.7. The whole set at the same
  number of steps was not run (dropped to keep the load on the machine down), so how much of the way from a
  quarter to the whole is left is measured only by the gap of section 3.

## Path forward, in the order of what it would buy

1. **Reuse every simulation at other scales** (section 5). An exact augmentation, free of simulation cost,
   that multiplies the set where it is thinnest. It needs a draw of (L, U) from the prior along the orbit of
   each stream and the track multiplied by L; the guards and the window prior scale with it. Making the mass
   ramp of dissolved progenitors relative would give them the second direction too.
2. **Simulate where the observed tracks are.** Long tracks come from 0.6 % of the prior. A second set drawn
   from a proposal that favours bound, multi-wrap orbits, with the prior ratio as loss weight, or a sequential
   round around the observed tracks, puts simulations where the effective sample is now 1.
3. **Resolve the orbits.** Steps set by the orbit (a fixed number per radial period, or 399 to 1,999 for the
   streams with several pericentres) instead of 99 for every stream. Until then `de_max: 0.01` restricts a run
   to resolved orbits; this keeps the likelihood smooth for importance and nested sampling but does not help
   the network.
4. **Mirror flip.** Checked on 2026-11-24 in StreaMAX-fit at `2e6a708`: the track files hold the sky angle
   `arctan2(y, x)`, sorted, and neither `export_strrings_tracks.py` nor the fit flips a track; the prior has
   v_y0 > 0 ("fixes the sense of rotation"). A stream that turns clockwise on the sky is therefore outside the
   model as given, and the 9 / 8 / 15 split of section 5 is what tracks with no preferred sense would give.
   The mirror y -> -y is an exact symmetry of the simulator: track angle phi -> -phi (bins reversed,
   `prog` -> 1 - `prog`), v_y0 -> -v_y0, `diry` -> -`diry`, everything else unchanged (not run as a numerical
   test). Two things to flip:
   - at inference, the observed track: run the network on the track and on its mirror image, and give the
     mirror result back in the sky frame with the signs of v_y0 and `diry` changed;
   - in training, the mock tracks: flip half of them and train a classifier of the sense on them, with logit
     f(track) - f(flipped track). It gives the weight of the two results, which the evidence from importance
     sampling cannot give for long tracks (efficiency about 1 %). The flow stays as it is, on v_y0 > 0.
5. **More particles for the long streams only.** The focused set of point 2 should use 90,000 particles
   (section 7): 1.6 to 3 times the usable samples for long tracks and the better-fitting ones among them, at
   three times the cost. The whole prior does not need it.
6. **Network**: a larger summary transformer (128 x 4 or more), with dropout or early stopping in mind, since
   the long tracks overfit; diffusion or flow matching in place of the coupling flow if the cost of their
   densities is acceptable. The run `npe_coupling_large_trainv2_1000epoch_vr4_5e4lr` (affine flow 12 x 1024,
   transformer 128 x 4, whole set, started 2026-11-24 18:04) tests the larger transformer at full length.
7. **Bin phase.** If the progenitor is not on a bin edge in the data, the mock tracks need bins at other
   phases, which needs the particles (section 6) or a finer stored grid.

## What was not done, and cautions

- Only NGC1162 has a finished nested-sampling run to compare with (section 2).
- The weighted-simulation posterior is only a reference where its effective sample is large; for the long
  observed tracks it rests on 1 to 5 simulations and its evidence is biased low.
- The scale augmentation was tested on the simulator only; it is in `mock.py` since 2026-11-25 (follow-up below).
- The orientation finding was not checked against how STRRINGS or StreaMAX-fit order the angle.
- The experiments ran once each, on a quarter of the set and a short schedule; differences below about
  0.05 nats are not established.
- `de_max` is in `data.py`, `predictive.py` and `track_noise.py`; `nested.py` does not know it.

## Follow-up of 2026-11-25: long streams at 90,000 particles, scale augmentation

Two items of the path forward are now in the code and in training.

- **`data/train_v2_long90k/`**: the 644,256 long streams of `train_v2` (9.09 % of the usable ones) simulated
  again with 90,000 particles. Its `README.md` has the rule, the checks and the numbers. With a 1 kpc data
  error the share of these streams with at least 26 usable bins rises from 30.7 % to 48.3 %. The run-file
  setting `replace_tracks` puts these tracks in place of the stored ones; guard 1 becomes 30 particles for
  them.
- **The selection is made on the 30,000-particle tracks**, so the particle number of a stream is a function
  of its parameters and of its first realisation. Importance sampling or nested sampling against a network
  trained on this set must do the same: simulate at 30,000 particles and, if the stream is long, again at
  90,000. The predictive check and the importance weights of `evaluation.py` and `evaluation_real.py` do
  this (`more_particles` in `sbi/predictive.py`); `prior_weights.py` and `nested.py` do not yet.
- **Scale augmentation** (`scale_aug` in the run file, `mock.py`): every training example is moved along the
  scale family of section 5, with (L, U) drawn from the prior restricted to that family (density of the
  scaled parameters times L^4 U in ln L, ln U; rederived, and tested on the whole set). A dissolved
  progenitor moves in L only. Validation mocks are never scaled. Tests and numbers:
  `scripts/extragalactic/scale_augmentation.py`, `plots/train_v2/scale_augmentation/scale_augmentation.json`.
- **Two runs**, started 2026-11-25 02:13 and 02:19, 1000 epochs, affine flow with the MLP summary, same seed,
  both on the new set: `npe_mlp_affine_long90k_1000epoch_vr4_5e4lr` (augmentation off) and
  `npe_mlp_affine_long90k_scaleaug_1000epoch_vr4_5e4lr` (on). When both end,
  `models/npe_mlp_affine_long90k_1000epoch_vr4_5e4lr/compare_when_done.sh` runs `nll.py --train` on them and
  writes `compare.log` there: loss on training and held-out streams by track length, and their gap.

## Files

- `figures/`: the figures named above, and `long_streams_census.png`, `long_streams_training.png` from
  `long_streams.py`.
- `numbers/`: `study.json` (figures 01, 03, 04, 07), `track_noise.json`, `prior_weights.json`,
  `symmetries.json`, `particle_svd.json`, `particle_store.json`, `long_streams.json`,
  `train_vs_heldout.txt` / `.npz`.
- `scripts/`: the one-off scripts behind `train_vs_heldout` and the importance-sampling tables.
- Code added in the repository (uncommitted): `scripts/extragalactic/track_noise.py`, `symmetries.py`,
  `particle_svd.py`, `particle_store.py`, `particle_number.py`, `accuracy_study.py`, `sbi/nll.py`,
  `sbi/prior_weights.py`; the setting `de_max` (`sbi/config.py`, `sbi/data.py`, `sbi/predictive.py`); the
  32-bit counts (`generate.py`, `sbi/mock.py`).
- `summary.html`: the short version of this file.
- `numbers/prior_weights_guard2_off.json`, `prior_weights_3x_particles.json`: section 4.
- `npe_v2_coupling_large.yaml`: the larger affine run started after the study.
- Runs: `models/q_*`, `models/s_data06`; `models/npe_trainv2_1000epoch_vr4_5e4lr/prior_weights*/`;
  `plots/train_v2/track_noise/`, `symmetries/`, `particle_svd/`; `data/particles_pilot/` (1.8 GB).
