"""Networks: a BayesFlow workflow built from the settings.

    inference   flow            coupling flow (exact density, cheap to sample); `transform`: spline | affine
                diffusion       diffusion model (density through the probability-flow ODE)
                flow_matching   flow matching
    backbone    mlp | transformer   subnet of diffusion / flow_matching; `transformer` is BayesFlow's
                                    DiffusionTransformer. The coupling flow always uses MLPs.
    summary     transformer     attention over the bins, the angle as explicit time axis, padding mask
                timeseries      convolutions + bidirectional GRU along the angle
                mlp             the track flattened into one vector

All three inference networks accept `fixed_target_mask`: a parameter marked as fixed is clamped to its given
value and left out of the density. That is how the progenitor position is passed when it is known (mock.py).
Only the DiffusionTransformer is told per coordinate what is fixed; the other subnets learn it from the
"position given" flag in the conditions.
"""

from __future__ import annotations

import bayesflow as bf
import keras
from bayesflow.networks.inference.coupling.transforms import spline_transform as _spline_transform

from config import Config

USES_MASK = {"transformer": True, "timeseries": False, "mlp": False}

_bayesflow_spline = _spline_transform._rational_quadratic_spline


def _guarded_spline(x, edges, derivatives, inverse: bool = False):
    """BayesFlow's rational-quadratic spline with a finite gradient for the points it is not used for.

    SplineTransform (BayesFlow 2.1.0) evaluates the spline for every point, in the first bin for a point outside
    the domain, and then takes the linear tail for those points with `where`. The value is right, the gradient
    need not be: right of the domain the spline's log Jacobian is log(sk^2 * derivative of the first inner knot),
    and when the subnet has pushed that derivative to exactly 0 (softplus of less than -104 in single precision)
    it is log(0). The discarded -inf has gradient 0 * inf = NaN, which reaches every weight upstream. One such
    row in 10^7 ended two training runs (2026-11-24). Here a point outside its bin is moved to the middle of the
    bin before the spline is evaluated. Points inside their bin, the only ones whose result is kept, are untouched,
    so no value changes.
    """
    lo, hi = (edges["bottom"], edges["top"]) if inverse else (edges["left"], edges["right"])
    x = keras.ops.where((x >= lo) & (x <= hi), x, 0.5 * (lo + hi))
    return _bayesflow_spline(x, edges=edges, derivatives=derivatives, inverse=inverse)


_spline_transform._rational_quadratic_spline = _guarded_spline  # SplineTransform.__init__ looks the name up there


@keras.saving.register_keras_serializable(package="streams")
class FlatMLP(bf.networks.SummaryNetwork):
    """Baseline summary network: the token grid flattened, then a residual MLP."""

    def __init__(self, summary_dim: int = 32, widths=(256, 256, 256), dropout: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.cfg = dict(summary_dim=summary_dim, widths=tuple(widths), dropout=dropout)
        self.mlp, self.head = bf.networks.MLP(widths=tuple(widths), dropout=dropout), keras.layers.Dense(summary_dim)

    def call(self, x, training: bool = False, **kwargs):
        return self.head(self.mlp(keras.ops.reshape(x, (keras.ops.shape(x)[0], -1)), training=training))

    def get_config(self):
        return super().get_config() | self.cfg


def network_names(cfg: Config) -> dict:
    """mock.py's name of every network input of a training batch -> BayesFlow's name."""
    names = dict(zeta="inference_variables", cond="inference_conditions", tokens="summary_variables", weight="sample_weight",
                 fixed_target_mask="fixed_target_mask")  # fmt: skip
    return names | (dict(mask="summary_mask") if USES_MASK[cfg.summary] else {})


def build_adapter(cfg: Config) -> bf.Adapter:
    ad, names = bf.Adapter().convert_dtype("float64", "float32"), network_names(cfg)
    for a, b in names.items():
        ad = ad.rename(a, b) if a != b else ad
    return ad.keep([*names.values(), "fixed_target_value"])


def build_summary(cfg: Config):
    if cfg.summary == "transformer":
        return bf.networks.TimeSeriesTransformer(summary_dim=cfg.summary_dim, embed_dims=(cfg.embed_dim,) * cfg.n_blocks,
                                                 num_heads=(cfg.n_heads,) * cfg.n_blocks, dropout=cfg.dropout, time_axis=0,
                                                 time_embedding="time2vec", time_embed_dim=16)  # fmt: skip
    if cfg.summary == "timeseries":
        return bf.networks.TimeSeriesNetwork(summary_dim=cfg.summary_dim, filters=(32, 64), kernel_sizes=(3, 3),
                                             recurrent_dim=cfg.embed_dim, dropout=cfg.dropout)  # fmt: skip
    return FlatMLP(cfg.summary_dim, (2 * cfg.embed_dim,) * cfg.n_blocks, cfg.dropout)


def build_inference(cfg: Config):
    if cfg.inference == "flow":
        if cfg.transform not in ("spline", "affine"):
            raise SystemExit(f"transform: {cfg.transform!r} is not spline or affine")
        # zeta is a standard normal a priori, within +-5: the spline domain covers it. The affine transform has no domain
        kw = dict(default_domain=(-5.0, 5.0, -5.0, 5.0)) if cfg.transform == "spline" else None
        return bf.networks.CouplingFlow(depth=cfg.depth, transform=cfg.transform, transform_kwargs=kw,
                                        subnet_kwargs=dict(widths=(cfg.width,) * cfg.flow_layers, dropout=cfg.dropout))  # fmt: skip
    subnet = dict(mlp="time_mlp", transformer="diffusion_transformer")[cfg.backbone]
    kw = dict(widths=(cfg.width,) * cfg.depth) | (dict(num_heads=cfg.n_heads) if cfg.backbone == "transformer" else {})
    net = dict(diffusion=bf.networks.DiffusionModel, flow_matching=bf.networks.FlowMatching)[cfg.inference]
    return net(subnet=subnet, subnet_kwargs=kw)


STANDARDIZABLE = ("inference_variables", "summary_variables", "inference_conditions")


def standardized(cfg: Config) -> list[str]:
    """The `standardize` setting -> the variables BayesFlow standardises. It keeps a running mean and sd of each
    feature (for the tokens over the observed bins only, through the summary mask), stores them in the model and
    applies them, and their inverse, in `sample` and `log_prob`; a given progenitor position is standardised too."""
    s = cfg.standardize.strip().lower()
    names = [] if s in ("", "none") else list(STANDARDIZABLE) if s == "all" else [x.strip() for x in s.split(",")]
    if set(names) - set(STANDARDIZABLE):
        raise SystemExit(f"standardize: {cfg.standardize!r} is not none, all or a list of {', '.join(STANDARDIZABLE)}")
    return names


def build_workflow(cfg: Config) -> bf.BasicWorkflow:
    """The workflow owns the estimator, the optimizer (AdamW, cosine decay with warm-up) and the checkpoint:
    `model.keras` in the run directory, rewritten at every epoch that lowers the held-out loss."""
    return bf.BasicWorkflow(adapter=build_adapter(cfg), inference_network=build_inference(cfg), summary_network=build_summary(cfg),
                            initial_learning_rate=cfg.lr, standardize=standardized(cfg) or None, checkpoint_filepath=str(cfg.run_dir()),
                            checkpoint_name="model", save_best_only=True)  # fmt: skip
