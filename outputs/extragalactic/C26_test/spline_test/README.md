# A smooth spline of the track, and the curvature test of arXiv:2606.21774 (2026-11-25)

Question (Giuseppe, 2026-11-25): can a smooth spline be fitted to an observed stream, with the unwrapped angle
as the independent variable; does the test of arXiv:2606.21774 (Euclid Q1, halo shapes from the curvature of
stream tracks) say anything on our tracks; and does it point to something better to do with the simulations
than the bins.

Code: `scripts/extragalactic/spline_track.py` (computations) and `spline_track_figures.py` (figures and
`numbers/summary.json`). CPU only, about two minutes on 16 cores:

```bash
uv run python scripts/extragalactic/spline_track.py all --workers 16
```

## Short answer

1. **Yes, a smooth spline describes every observed track.** At the errors of the data a track holds 3 to 20
   effective numbers, 0.37 per bin (median), and mock tracks cut from the simulations hold as many.
2. **The paper's test cannot tell the halo shape within our prior.** For halos with q between 0.5 and 1.5 the
   acceleration projected on the sky points within 10 degrees of the host centre (90th percentile), so the
   test reduces to "the track must bend towards the host", whatever q is. Simulated streams break that rule
   in 7 % of their bins, most of them next to the progenitor, at every q alike.
3. **The spline is a useful check of the data against the model.** Two long tracks, NGC1162 and NGC5263, are
   smooth to 0.2 kpc, yet the stream model needs 1.6 to 2.1 kpc of extra scatter to fit them (nested sampling
   of NGC1162: 1.9 kpc). Both have a stretch that bends away from the host, as clearly as at most 3.5 % of the
   mock tracks show one. For the other 30 tracks the stream model fits as well as a free smooth curve.
4. **For the simulations, a spline should not replace the bins.** A simulated track has real structure from
   bin to bin, above its particle noise: a spline with a knot every two or three bins is 2 to 5 times worse
   than the bins. A penalised spline with a knot per bin, one for each arm of the stream, lowers the squared
   error by a quarter, which is worth 1.3 times the particles (90,000 particles are worth 3 times).
5. **A spline does give the track at another bin phase, except next to the progenitor.** From 30 degrees on,
   the track in bins shifted by half a bin is predicted from the stored bins as well as binning the particles
   again does. Within two bins of the progenitor it is off by 4 to 7 % of the radius (90th percentile), which
   is the size of the observed errors.

## What was done

- **Spline.** r(phi) as a penalised cubic B-spline, phi the unwrapped angle of the track files, one knot per
  10 degree bin, penalty on the third differences of the coefficients (changes of r''; the stiff limit is a
  parabola). Errors are Gaussian with variance sigma_i^2 + s^2, s a scatter beyond the stated errors as in the
  likelihood of Chemaly et al. 2026b. The smoothing parameter and s are not chosen but given a posterior on a
  grid (restricted marginal likelihood, flat prior in log lambda and in s up to 5 kpc), and 400 curves are
  drawn from it. This stands in for the optimiser and NUTS of the paper; the model is linear, so the draws
  are exact.
- **Why the angle and not the arc length.** The paper fits a curve drawn in the image, in a normalised arc
  length. Our files hold the binned track only (angle, radius, error), and the simulated track is binned in
  the same angle, so r(phi) is single-valued by construction and the angle is the natural variable.
- **Curvature.** With x = r cos phi, y = r sin phi the sign of the curvature is the sign of
  r^2 + 2 r'^2 - r r''. Positive: the track bends towards the side the host is on. Negative: it bends away.
- **The paper's likelihood** (its eq. 6 to 9): the share of spline draws whose curvature vector lies within
  90 degrees of the halo's acceleration at every one of I points evenly spaced in arc length (I = number of
  bins). Halo of the paper: logarithmic potential flattened by q in the plane of the sky along a position
  angle; grid q = 0.10 to 1.00, position angle every 5 degrees.
- **Mock tracks.** For every observed track, 200 mock tracks with its bins, gaps and errors, cut from
  simulated streams around their progenitor, radii scaled to the median radius of the track (the simulator's
  length scaling), both guards applied (10 particles per bin of 30,000, `var_ratio` 4), no scatter beyond the
  errors added. Two
  pools: `prior` (the 5,000 prior draws of `data/particles_pilot`, 30,000 particles) and `long` (the 7,871
  long or well-fitting streams of `plots/train_v2/particle_number`, 90,000 particles). The same analysis is
  run on them.
- **Simulations with 300,000 particles** (1,797 of the streams of `particle_number`, the same parameters at
  30,000 particles): the track against the true halo, and the 30,000-particle track smoothed by a spline
  against the 300,000-particle one.
- **Particles** (`data/particles_pilot`, 2,728 streams with a run of 8 bins or more): the particles of a
  stream are split at random in two halves; from the standard bins of one half the track of the other half is
  predicted in bins shifted by half a bin.

## 1. The observed tracks as splines

Figures `1_observed_tracks_p*.png` (radius against angle), `2_observed_sky_p*.png` (plane of the track
files), `3_observed_vs_mocks.png`. Numbers per track in `numbers/observed.json`.

- **Effective number of parameters** (trace of the smoother): 3.2 to 20.4; 0.21 to 0.62 per bin, median 0.37.
  16 of the 32 tracks hold 6 or fewer, 24 fewer than 10; only PGC938075 (15.2), PGC1001085 (17.0) and
  UGC08717 (20.4) hold more than the 14 parameters of the stream model.
- **Against mock tracks** the numbers agree: three tracks are above 97.5 % of the `prior` mocks (UGC08717,
  ESO079-003, PGC006791), none against the `long` pool, where 1 to 2 of 32 are expected by chance.
- **Scatter beyond the stated errors.** The posterior median of s is above 97.5 % of the mocks for NGC4256
  (3.1 kpc), NGC5513 (3.2) and UGC08717 (1.5), and above 95 % for seven tracks in the `prior` pool (those,
  UGC01970, IC1657, NGC5055, UGC01424) where 1.6 are expected. These tracks jump from bin to bin by more than
  their errors, which no smooth curve and no stream model can follow; `sigma_sys` is needed for them.
  The value of s alone means little for a short track: mocks made with no extra scatter give a median s of
  0.2 to 1.9 kpc, because a few bins cannot exclude it. Only the comparison with the mocks counts.

## 2. The curvature test of the paper

Figures `4_halo_shape.png`, `5_premise.png`. Numbers in `numbers/summary.json` (`shape`, `premise`).

**On the observed tracks**

- No track has a long stretch that bends away from the host: the share of the length where that has a
  posterior probability above 95 % is 3 % for NGC5263, 1 % for NGC0804 and zero for the other 30.
- Four tracks have a place where it is above 90 %: NGC5263 (0.998, at angle 350 deg, radius 9 kpc, the
  innermost part of the track), NGC0804 (0.95, at 340 deg, 16 kpc, also its innermost part), NGC1162 (0.94,
  at 222 deg, near one end) and UGC01424 (0.94, at 430 deg, 7 kpc). Against the mocks: none of 400 mocks
  reaches the value of NGC5263; 1 % reach UGC01424's, 1.5 to 3.5 % NGC1162's, 3 to 7.5 % NGC0804's.
- **The paper's likelihood is empty for the long tracks.** A spherical halo centred on the host is allowed by
  no spline draw of 7 tracks, all of 22 bins or more, and for 8 tracks the likelihood of the best q, averaged
  over the position angle, is below 5 draws in 400. This is the product over 22 to 48 points: a draw fails if
  its curvature has the wrong sign at one of them. 18 % of the mock tracks have no allowed draw at all. The paper's splines
  are far stiffer (a knot per 50 pixels and a cost on changes of curvature).
- **The stack of the other 24 tracks prefers a round halo**: maximum at q = 0.95, flat from 0.8 to 1, and
  2.3 lower in log-likelihood at q = 0.5. The paper reports q = 0.95 (+0.05, -0.10) for its 13 galaxies.
- **The same preference comes out of mock tracks whatever the true flattening.** The round halo is the best
  one, or tied, for 96 % of the mocks, and the likelihood at q = 0.5 relative to the best is 0.93, 0.87 and
  0.92 for true flattenings of 0.5-0.7, 0.9-1.1 and 1.3-1.5. So on our tracks this likelihood does not measure q.

**In the forward model** (300,000 particles, 702 thin streams with a resolved orbit, 32,136 bins)

- The halo's acceleration, projected on the sky at the three-dimensional place of the stream, points within
  0.4 degrees (median) of the host centre for q of 0.9-1.1 and within 3.9 degrees for q of 0.5-0.7; the 90th
  percentile is 10.6 degrees at most, the 99th 23 degrees. It points to the far side of the track in 0.02 %
  of the bins. For every halo of the prior the paper's criterion is therefore "the track bends towards the
  host".
- The tracks do not obey it: 6.7 % of the bins bend away significantly (59 % bend towards the host, 34 % are
  undetermined), and 90 % of the streams have such a bin. The share is 5.9, 7.0, 7.4, 6.3 and 6.7 % in five
  bins of q from 0.5 to 1.5: no dependence on the halo shape.
- Half of these bins are within 30 degrees of the progenitor: 37 % of the bins within 10 degrees bend away,
  31 % at 10 to 20 degrees, 10 % at 20 to 30 degrees and 2 to 4 % beyond 90 degrees. It is the same for a
  surviving and for a dissolved progenitor (24 % and 27 % within 30 degrees). A stream is not an orbit, least
  of all where its two arms meet.
- Streams with an unresolved orbit bend away in 17 % of their bins.
- These are significances at the particle noise of 300,000 particles, far below the errors of the data. With
  the bins and errors of the observed tracks the mocks of section 1 are the measure.

## 3. What the stream model does not fit of a smooth track

Figure `8_scatter_model_vs_spline.png`, numbers under `scatter` in `numbers/summary.json`.

The scatter beyond the errors that the stream model asks for (`sig` of the network
`npe_trainv2_1000epoch_vr4_5e4lr`, 1,000 posterior draws per track) against the scatter s the spline leaves.

- For 30 tracks the two agree within their uncertainties: the stream model fits them as well as a free
  smooth curve.
- **NGC1162**: spline 0.15 kpc (0.03 to 0.37), network 1.63 (1.15 to 2.11), nested sampling 1.88 (1.42 to
  2.35). **NGC5263**: spline 0.22 (0.05 to 0.52), network 2.05 (1.40 to 2.78). Both tracks have 28 bins,
  errors of 2.4 and 4.4 % of the radius, and are smooth within them; the stream model misses their shape by
  ten times what a smooth curve does. They are two of the four tracks with a bend away (section 2), and two
  of the four the accuracy study found the best simulations to miss "by a few sigma along whole sections".
- For PGC039258 and NGC5055 the network asks for less scatter than the spline leaves (0.6 against 2.2 kpc,
  0.7 against 1.4). Either the stream model follows a sharp feature that one smoothing parameter cannot, or
  the network's posterior is off there; not looked into.

## 4. The spline as the model track

Figure `6_smoothing.png`, numbers under `smoothing`. Thin streams, 30,892 bins; the squared error of a model
track against the 300,000-particle track of the same parameters, in units of the particle noise of 30,000
particles (variance of the bin median, 1.25^2 hw^2 / n), noise of the reference removed.

| model track from 30,000 particles | squared error | error over its own estimate | long streams with 26 usable bins at 1 kpc |
|---|---:|---:|---:|
| bins as stored | 1.05 | | 30 % |
| one penalised spline, knot every bin | 0.91 | 1.04 | 36 % |
| one per arm, knot every bin | 0.81 | 1.07 | 41 % |
| one per arm, knot every 2 bins | 2.04 | 1.62 | 48 % (error underestimated) |
| one per arm, knot every 3 bins | 4.94 | 2.39 | 57 % (error underestimated) |
| bins, 90,000 particles | 0.33 expected | | 52 % |
| bins, 300,000 particles | 0.10 expected | | 68 % |

- **The simulated track is not smooth at the level of its particle noise.** The penalised spline chooses
  almost no smoothing (0.86 effective parameters per bin), and a spline with fewer knots than bins is biased
  by more than the noise it removes.
- **One spline per arm** (split at the progenitor) is the best of them: 23 % less squared error, as 1.3
  times the particles would give, with an honest error estimate. Used in the particle-noise guard it lets 41 %
  of the 603 long streams keep 26 usable bins at a data error of 1 kpc, against 30 % now and 52 % with 90,000
  particles (0.3 kpc: 7 % against 5 % and 14 %; 3 kpc: 73 % against 66 % and 79 %).
- Next to the progenitor (within 30 degrees) one spline across it is worse than the bins (0.99 against 0.76):
  the two arms meet there with a step.
- In kpc: the median error of the stored bins is 0.39 % of the radius, and 0.33 % with the per-arm spline;
  the observed errors are 4.1 % of the radius (median; 1.3 % at the 10th percentile).

## 5. The spline as the track at another bin phase

Figure `7_bin_phase.png`, numbers under `phase`. Thin streams of the particle set (1,234), halves of 15,000
particles; errors in units of the particle noise of the predicted bin. "Floor" is the half binned again at
the new phase.

| shifted bin | floor: median, 90th pct | mean of the neighbours | one penalised spline | one spline per arm |
|---|---|---|---|---|
| across the progenitor | 0.5, 2.5 | 7.9, 31 | 22, 107 | not defined |
| 10 deg from it | 0.9, 2.3 | 6.2, 21 | 9.0, 26 | 4.4, 15 |
| 20 to 30 deg | 1.0, 2.4 | 1.9, 5.2 | 1.8, 5.9 | 1.4, 4.0 |
| 30 to 40 deg | 1.0, 2.4 | 1.3, 3.5 | 1.0, 2.7 | 1.0, 2.6 |
| beyond 40 deg | 1.0, 2.3-2.5 | 1.0-1.1, 2.7-3.0 | 0.9, 2.3-2.5 | 0.9, 2.3-2.4 |

- From 30 degrees on, a spline of the stored bins gives the shifted track at the floor. The mean of the two
  neighbours, which the accuracy study tried, is 10 to 30 % worse.
- At the progenitor nothing works: the bin across it mixes the two arms, and the bins next to it are off by
  4 to 9 times the noise in the median. In kpc the 90th percentile of the error is 4 to 7 % of the radius there, against
  0.5 to 0.7 % for the floor.
- The 90th percentile over all bins of a stream is dominated by these few bins (8 to 12 times the noise for
  every interpolation of the whole stream; 2.6 for the per-arm spline from 20 degrees on, with a 99th
  percentile of 8 against 3.8 for the floor).

## What this says about the simulations

1. **Keep the bins as the model track.** A stiffer spline loses real structure; the particles are what
   lowers the noise. The per-arm penalised spline is a free gain of a quarter in squared error and could
   relax the particle-noise guard a little (30 to 41 % of long streams with 26 usable bins at 1 kpc), at the
   price of a guard that is no longer the paper's. Not put in the code.
2. **Bin phase** (point 7 of the accuracy study). Mock tracks at any bin phase can be made from the stored
   bins with a per-arm spline, without particles, for bins 30 degrees or more from the progenitor. The two or
   three bins on each side of it cannot, and there the phase matters at the level of the observed errors.
   A finer stored grid near the progenitor only (for example 2.5 degree bins within 30 degrees of it) would
   close this at little storage. Suggested, not tested.
3. **Treat the two arms separately at the progenitor.** Anything smooth across it (a spline, a neighbour
   average, a curvature) is wrong there.
4. **The sign of the curvature is not a summary worth adding for the halo shape.** Within the prior it does
   not depend on q.
5. **The spline is worth keeping as a check of each observed track**: the scatter it leaves against the
   `sigma_sys` of the stream model shows which tracks the model cannot make (NGC1162 and NGC5263 now), with
   no simulation.
6. **Spline coefficients as network input were not tried.** The bins are about three times redundant at the
   errors of the data, but the mocks have the same redundancy and the summary network already compresses
   them, so no gain is expected from it.

## Cautions

- The paper was read from its arXiv abstract and HTML pages; its appendix B (cost weights of the spline) was
  not. Its method differs from this one in the points listed under "What was done".
- The errors of neighbouring bins are taken as independent. If the observed ones are correlated, the tracks
  look smoother than they are and s is too low.
- Mock tracks always hold the progenitor inside the window and have no scatter beyond their errors. The
  `long` pool is a picked set, not the prior. With 200 mocks a probability below 0.5 % cannot be told from
  zero, and with 32 tracks 1 to 2 are expected above 97.5 % of the mocks in any one statistic.
- The network's `sig` for a long track comes from a posterior that importance sampling does not confirm
  (accuracy study, section 2); for NGC1162 nested sampling gives the same answer, for NGC5263 there is no
  such check.
- The position angle of the paper's halo is in the frame of the track files, whose orientation on the sky is
  not known (CLAUDE.md), so it is not reported.
- Where the four bends away come from (the progenitor, a second wrap, something outside the model) was not
  looked into. For NGC1162 the bend at 222 degrees is not where nested sampling puts the progenitor (291
  degrees, 260 to 374).

## Files

- `figures/`: `1_observed_tracks_p1/p2.png`, `2_observed_sky_p1/p2.png`, `3_observed_vs_mocks.png`,
  `4_halo_shape.png`, `5_premise.png`, `6_smoothing.png`, `7_bin_phase.png`, `8_scatter_model_vs_spline.png`.
- `observed.npz`: per track `<name>/x, y, sig` (data), `knots, beta` (spline: knots and posterior-mean
  coefficients, `scipy.interpolate.BSpline(knots, beta, 3)` in degrees), `xg, r, r_lo, r_hi, p_away` (mean
  curve, 16-84 % band and probability of bending away on a grid of 2.5 degrees), `post` (posterior of s and
  lambda on `s_grid` x `lam_grid`), `like` (paper's likelihood on `q_grid` x `pa_grid`), `L_q`, `xi`.
- `numbers/observed.json`: one row per track. `numbers/summary.json`: every number quoted here, and
  `mock_table` (observed value, mock percentiles and tail probabilities per track, pool and statistic).
  `numbers/mocks_meta.json`: streams and acceptance per track and pool.
- `mocks.npz`, `premise.npz`, `smoothing.npz`, `phase.npz`: the tables behind sections 1, 2, 4 and 5
  (columns named in each file).
