"""The experiment config tree.

An experiment *is* a config file (PLAN.md invariant 6), so every knob that changes what is
being measured lives here and every result carries :meth:`Config.config_hash`.

House convention (PLAN.md §13, density reference ``../RGBT-Fusion-Detection/src/rgbt/config.py``):
**every field carries an inline comment naming its failure mode.** A knob whose failure
mode nobody can state is a knob nobody should be turning.

Paths are never checked for existence here. Configs are authored on the Mac and resolved on
the training server, so a path that is missing locally is the normal case; existence is a
startup check in the engine layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import Field, ValidationError, field_validator

from t2o.config.base import ConfigBase, ConfigError, as_config_error, deep_merge
from t2o.imaging import Normalize

CONFIG_FILENAME = "config.yaml"


class AmpDtype(StrEnum):
    """Autocast dtype. bfloat16 needs no GradScaler on Ampere+; float16 does."""

    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class Backbone(StrEnum):
    """Translator backbone, and the discriminator for :data:`TranslatorConfig`.

    Members are added as each backbone lands: ``pix2pix`` at M1, ``pix2pix_turbo`` at M2a,
    ``lbbdm`` at M2b (PLAN.md §4).
    """

    STUB = "stub"
    PIX2PIX = "pix2pix"
    PIX2PIX_TURBO = "pix2pix_turbo"


class DataConfig(ConfigBase):
    # (path) the dataset's own data.yaml -- it declares root, splits, nc, names and the
    # modality tokens. Nothing about the dataset is duplicated into this config, so the
    # same experiment config runs against any dataset. See t2o.data.manifest (M0.3).
    manifest: Path = Path("dataset/yolo_rgbt/data.yaml")
    # (float) fraction of training annotations kept, for the E8 low-annotation sweep. 1.0
    # is the full set. Subsampling is seeded and deterministic, so two runs at the same
    # fraction and seed see the same subset; changing this without changing the seed
    # yields a nested subset rather than an independent draw.
    annotation_fraction: float = 1.0
    # (int) seed for that subsample only, held separate from train.seed so an E8 sweep can
    # vary model initialisation without also redrawing which images are labelled.
    annotation_seed: int = 0

    @field_validator("annotation_fraction")
    @classmethod
    def _fraction_in_unit_range(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("must be in (0, 1]; 0 would leave nothing to train on")
        return value


class TrainConfig(ConfigBase):
    # (int) translator epochs within each lambda_det stage.
    epochs_per_stage: int = 10
    # (int) images per step. Raise for throughput until VRAM says no; the detection loss is
    # divided by it (PLAN.md §8), so lambda_det does not silently scale with this.
    batch_size: int = 8
    # (float) Adam LR for the translator.
    lr: float = 1.0e-4
    # (float) per-epoch ExponentialLR decay. 1.0 disables decay.
    lr_gamma: float = 0.95
    # (bool) autocast. Off costs throughput; on with float16 and no GradScaler produces
    # silent NaNs, which is why amp_dtype defaults to bfloat16.
    amp: bool = True
    amp_dtype: AmpDtype = AmpDtype.BFLOAT16
    # (float) probability of a horizontal flip. Geometric augmentation only: photometric
    # jitter would move the visible target the reconstruction loss scores against.
    hflip: float = 0.5
    # (ints|null) random crop [height, width], or null for full frames. Must stay
    # stride-32 divisible or the in-loop detector rejects it.
    crop: tuple[int, int] | None = None
    # (int) seed for init, shuffling and augmentation. Part of experiment identity and so
    # part of config_hash -- resuming under a different seed is drift worth catching.
    seed: int = 0

    @field_validator("epochs_per_stage", "batch_size")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("hflip")
    @classmethod
    def _is_a_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must be a probability in [0, 1]")
        return value


class LossConfig(ConfigBase):
    """The three fidelity terms of PLAN.md §8's

    ``lambda_l2*L2 + lambda_lpips*LPIPS + lambda_gan*GAN + lambda_det*Detection``.

    The fourth term's weight is a schedule, not a scalar, and lives in
    :class:`CouplingConfig`.
    """

    # (float) pixel L2 against the real visible frame. The reason a translator with no
    # other term produces a blurred conditional mean -- but *not* dominant at these
    # weights: measured over E3's campaign it is 1.0% of the objective, against LPIPS at
    # 44% and GAN at 52% (TASKS.md M1.2 step 7). Raising `lpips`/`gan` is what made it
    # small; the scale of the raw MSE on [0,1] images (~0.03) is the other half.
    l2: float = 1.0
    # (float) LPIPS perceptual term -- what buys back the texture L2 blurs away. Also the
    # quantity the fidelity floor watches (PLAN.md §8), so zeroing it disables that
    # guardrail as a side effect.
    lpips: float = 5.0
    # (float) adversarial term. 0.0 trains without a discriminator, which is the cheaper
    # and more stable starting point; raise it once the loop itself is working.
    gan: float = 0.0


class CouplingConfig(ConfigBase):
    """Detection-in-the-loop coupling: the C1 contribution's only knobs."""

    # (floats) the staged lambda_det ramp, one entry per stage. task_weights[0] == 0 keeps
    # stage 0 translation-only, so no detector is constructed to start. A single-element
    # [0.0] is the clean no-loop control that makes E3 a genuine ablation -- at weight 0
    # the frozen detector is never built at all (PLAN.md §8).
    task_weights: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
    # (float) constant downscale on the detection gradient before it reaches the
    # translator. The reward-tuning literature runs this small: ReFL 1e-3, AlignProp 1e-2.
    # Too high and the detector's gradient overwhelms fidelity, producing adversarial
    # texture that scores well and looks wrong. Too low and the term does nothing at all,
    # which is the failure that actually happened: this default put the detection term at
    # 2.3% of the objective in E3, and the campaign's null was uninterpretable as a
    # result. Calibrate it per backbone with `scripts/loss_share.py` -- the effective
    # weight is `task_weights[stage] * grad_scale`, so this is not read alone.
    grad_scale: float = 1.0e-2
    # (float|null) saturating-reward target: once the detection loss falls below this the
    # sample stops being rewarded, which blunts reward hacking directly (ReFL's
    # `relu(-r + 2)`, AlignProp's `|r - target|`). null restores unbounded minimisation,
    # which is the configuration that invites the detector to find adversarial textures.
    reward_target: float | None = None

    @field_validator("task_weights")
    @classmethod
    def _at_least_one_stage(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("must contain at least one stage")
        if any(weight < 0.0 for weight in value):
            raise ValueError("weights must be non-negative")
        return value


class InLoopDetectorConfig(ConfigBase):
    """The frozen detector that guides training and receives generator gradients.

    Never trained, never shares weights with :class:`EvalDetectorConfig` -- PLAN.md
    invariant 7, encoded structurally so the two cannot be conflated by accident.
    """

    # (path) frozen weights. An undertrained detector contributes gradient noise rather
    # than semantic signal, so COCO yolo11n.pt is a poor choice here; point it at the
    # optical detector already trained on this domain.
    weights: Path = Path("yolo11n.pt")
    # (int) must be stride-32 divisible, or v8DetectionLoss's feature maps do not line up
    # with the image. 640x480 sources are (20x32, 15x32).
    imgsz: int = 640

    @field_validator("imgsz")
    @classmethod
    def _stride_divisible(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError(f"must be divisible by the coarsest stride 32, got {value}")
        return value


class EvalDetectorConfig(ConfigBase):
    """The detector trained from scratch on exported translations, then measured.

    Independent of the in-loop detector by construction: it never sees a gradient from the
    translator, which is what keeps the reported mAP an honest number.
    """

    # (path) bootstrap weights for the *evaluation* detector. Must be identical across
    # every arm of the comparison or the results table stops being a comparison.
    init_weights: Path = Path("yolo11n.pt")
    # (int) epochs per stage. Too few and mAP differences between arms are swamped by
    # under-training noise rather than reflecting translation quality.
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16


class ReferenceDetectorConfig(ConfigBase):
    """The fixed, never-trained detector that scores translations *zero-shot*.

    A third role, distinct from both of the above. :class:`EvalDetectorConfig`'s detector is
    fine-tuned on each stage's translated images before it is measured, so its mAP is an
    in-domain, post-adaptation number -- and M0.10's E1 bracket showed that number saturates
    near 0.9 for *any* consistent domain on this dataset, including raw thermal. It therefore
    cannot answer M1's gate ("does translation beat raw thermal"), which is a question about
    an **unadapted** detector: the same visible-trained weights E1 measured at < 0.05 mAP50 on
    raw thermal, run on translated images instead. That is what this detector is for.
    """

    # (path|null) null falls back to detector.evaluation.init_weights, which is the right
    # detector when no independently-trained one exists yet. Resolving to the same file as
    # detector.in_loop.weights makes every lambda_det > 0 stage self-graded -- the loop warns
    # rather than failing, because the lambda_det = 0 arm stays clean either way.
    weights: Path | None = None
    # (int) validation resolution. Diverging from evaluation.imgsz would compare the two arms
    # at different scales, so keep them equal unless something specifically demands otherwise.
    imgsz: int = 640
    # (int) validation batch. Throughput only -- mAP is batch-invariant here, unlike export.
    batch: int = 16


class DetectorConfig(ConfigBase):
    in_loop: InLoopDetectorConfig = InLoopDetectorConfig()
    evaluation: EvalDetectorConfig = EvalDetectorConfig()
    reference: ReferenceDetectorConfig = ReferenceDetectorConfig()


class StubTranslatorConfig(ConfigBase):
    """The CPU stand-in translator (M0.6).

    Load-bearing, not a test fixture: it is the only backbone small enough to run the full
    data -> coupling -> export -> eval path in seconds on a CPU, which is what keeps that
    path locally testable at all (PLAN.md §9).
    """

    backbone: Literal[Backbone.STUB] = Backbone.STUB
    # (int) width of the stand-in's single hidden conv. Exists to be small; raising it
    # buys nothing but slower smoke tests.
    hidden_channels: int = 16


class Pix2PixTranslatorConfig(ConfigBase):
    """M1's GAN backbone (`PLAN.md` §10 Phase 1) -- vendored `models/networks.py` from
    `junyanz/pytorch-CycleGAN-and-pix2pix` at `2a7afba`, driven by our own trainer.

    Only the architecture knobs worth sweeping are exposed. `norm="batch"`, `init_type=
    "normal"`, `init_gain=0.02`, `beta1=0.5`, `n_layers_d=3` and the LPIPS backbone used for
    training loss are fixed inside `translators/pix2pix.py` -- nothing in Phase 1 needs to
    sweep them, and every config field carries an ongoing documentation cost (house style,
    PLAN.md §13). Reconstruction/adversarial loss weights are *not* here -- pix2pix reuses
    the shared `LossConfig.{l2,lpips,gan}` fields `StubTranslator` already established,
    rather than inventing its own `lambda_L1`.
    """

    backbone: Literal[Backbone.PIX2PIX] = Backbone.PIX2PIX
    # (resnet_9blocks|resnet_6blocks|unet_128|unet_256) generator architecture. Deliberately
    # *not* the paper's own default (`unet_256`): `UnetGenerator` needs input divisible by
    # 2**8=256, which neither the 640x480 custom dataset nor any stride-32-divisible
    # `train.crop` satisfies. `ResnetGenerator` only needs divisible-by-4.
    net_g: Literal["resnet_9blocks", "resnet_6blocks", "unet_128", "unet_256"] = "resnet_9blocks"
    # (basic|n_layers|pixel) discriminator architecture. "basic" is the paper's own default
    # (70x70 PatchGAN) and works on any input size.
    net_d: Literal["basic", "n_layers", "pixel"] = "basic"
    # (int) generator's last-conv filter width. Raising it costs VRAM/throughput for
    # (usually marginal) capacity.
    ngf: int = 64
    # (int) discriminator's first-conv filter width. Only matters when `loss.gan > 0` --
    # otherwise no discriminator is ever constructed (the same "clean no-op at weight 0"
    # discipline `CouplingConfig` already uses for the frozen detector).
    ndf: int = 64
    # (vanilla|lsgan) adversarial objective. Deliberately excludes the paper's third option,
    # `wgangp` -- that needs `cal_gradient_penalty` wiring this milestone doesn't build, so
    # it is rejected at config-load time rather than silently behaving like a no-op.
    gan_mode: Literal["vanilla", "lsgan"] = "vanilla"

    @field_validator("ngf", "ndf")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class Pix2PixTurboTranslatorConfig(ConfigBase):
    """M2a's one-step diffusion backbone (`PLAN.md` §4, §10 Phase 2a) -- sd-turbo with LoRA
    adapters, vendored from `GaParmar/img2img-turbo` at `463b2d3`.

    Same discipline as the pix2pix member: only knobs worth sweeping appear. The fixed
    timestep (999), the LoRA target-module lists, the padding multiple, AdamW's betas and the
    gradient-clip norm live in `translators/pix2pix_turbo.py`. Loss weights come from the
    shared `LossConfig.{l2,lpips,gan}` -- the adversarial term reuses M1's already-vendored
    PatchGAN rather than upstream's `vision_aided_loss`, so a backbone comparison changes the
    backbone and nothing else.
    """

    backbone: Literal[Backbone.PIX2PIX_TURBO] = Backbone.PIX2PIX_TURBO
    # (str) HuggingFace repo id for the frozen base. Downloaded to the HF cache on first use,
    # never into this repo. Changing it changes every number the run produces, which is why
    # it is an experiment field and not a runtime one.
    pretrained: str = "stabilityai/sd-turbo"
    # (str) the constant caption. Encoded once at construction, so it costs nothing per step
    # -- but it conditions every generated image, so it is part of experiment identity.
    prompt: str = "a photo of overhead power line components in daylight"
    # (int) LoRA rank on the UNet. Upstream's default is 8. Lowering it is the primary
    # anti-reward-hacking lever (PLAN.md §4): less capacity to distort the image in whatever
    # direction the detector happens to reward.
    lora_rank_unet: int = 8
    # (int) LoRA rank on the VAE. Upstream's default is 4 -- deliberately below the UNet's,
    # since the VAE path also carries the skip connections that preserve input structure.
    lora_rank_vae: int = 4
    # (basic|n_layers|pixel) discriminator architecture, only built when `loss.gan > 0`.
    # Same field as the pix2pix member on purpose: it is the same vendored PatchGAN.
    net_d: Literal["basic", "n_layers", "pixel"] = "basic"
    # (int) discriminator's first-conv filter width. Inert at `loss.gan == 0`.
    ndf: int = 64
    # (vanilla|lsgan) adversarial objective; `wgangp` is excluded for the same reason as in
    # the pix2pix member -- the gradient-penalty wiring does not exist.
    gan_mode: Literal["vanilla", "lsgan"] = "vanilla"

    @field_validator("lora_rank_unet", "lora_rank_vae", "ndf")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank; the caption conditions every generated image")
        return value


# A discriminated union: each new backbone adds a model with its own `backbone: Literal[...]`
# tag plus one entry here; nothing else in the config layer changes, which is what makes
# PLAN.md invariant 3 ("swapping backbones is a config change") true rather than aspirational.
TranslatorConfig = Annotated[
    StubTranslatorConfig | Pix2PixTranslatorConfig | Pix2PixTurboTranslatorConfig,
    Field(discriminator="backbone"),
]


class MetricsConfig(ConfigBase):
    """Knobs for :mod:`t2o.metrics` that change the reported number, not just its cost."""

    # (alex|vgg|squeeze) LPIPS backbone. Changes the reported number -- part of experiment
    # identity, not an implementation detail. 'alex' is the standard choice for evaluation
    # (lower variance, closer to the LPIPS paper's human-judgement calibration).
    lpips_net: Literal["alex", "vgg", "squeeze"] = "alex"
    # (int) KID's own default (1000) raises if fewer images than that are available. Both
    # the synthetic smoke fixture (a handful of images) and the ~850-pair custom dataset
    # are far below it; FidelityEvaluator additionally clamps at compute time if even this
    # is too high for a given run.
    kid_subset_size: int = 50
    # (float) IoU at or above which a predicted box counts as matching a reference box, for
    # false-object/missed-object rate and detection-consistency (faithfulness.py). Kept at
    # mAP50's own convention so the two kinds of number stay comparable.
    iou_threshold: float = 0.5
    # (float) minimum detector confidence for a prediction to be considered at all, for the
    # same three rates -- ultralytics' own inference default. Too low and false-object rate
    # is dominated by detections a real deployment would already discard, not by anything
    # the translator did.
    conf_threshold: float = 0.25

    @field_validator("iou_threshold", "conf_threshold")
    @classmethod
    def _is_a_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must be in [0, 1]")
        return value


class ExportConfig(ConfigBase):
    # (clamp|per_image) how translated floats map to uint8 on export. `clamp` keeps
    # intensities comparable across the dataset, which is what the evaluation detector
    # needs; `per_image` stretches each frame independently and makes them incomparable.
    # Never batch-wise -- that makes exported pixels depend on batch composition.
    normalize: Normalize = Normalize.CLAMP


class RuntimeConfig(ConfigBase):
    """Invocation details. **Excluded from config_hash in its entirety.**

    Nothing here changes what is being measured, so the same experiment run on two GPUs
    under two names must carry the same hash. The seed lives in :class:`TrainConfig` for
    exactly that reason -- it is scientific, these are not.
    """

    # (str|null) null auto-selects. "cpu", "cuda:0". No DDP: NCCL does not exist on
    # native Windows, so the second A100 is throughput, not scale (PLAN.md §3).
    device: str | None = None
    # (int) dataloader workers. Held at 0 until proven, since Windows spawns rather than
    # forks (PLAN.md §3); up to 16 now confirmed clean on the server. Belongs here, not in
    # TrainConfig, only because augmentation draws from a per-sample generator keyed on
    # (seed, epoch, index) -- see data/dataset.py. Change that and this silently becomes
    # experiment identity again while sitting outside config_hash.
    workers: int = 4
    # (bool) self-hosted W&B logging. The base URL and API key come from the server
    # environment; a key in this file would be committed to a public remote.
    wandb: bool = False
    # (str) project every run lands under. Kept separate from `name` so runs are
    # comparable to each other instead of each creating its own project.
    wandb_project: str = "thermal-to-optical"
    # (path) parent of all run outputs. Ignored by git -- results must not travel back
    # through the repo (PLAN.md §5).
    run_dir: Path = Path("runs")
    # (str) this run's directory name. Reusing a name resumes into that directory, which
    # is deliberate but means a copy-pasted name silently continues someone else's run.
    name: str = "t2o"
    # (str|null) W&B group. E3's campaign is 12 runs that only mean anything read against
    # each other; ungrouped they are unnavigable siblings in the project list. null leaves
    # a run ungrouped. Purely organisational, hence here rather than anywhere hashed --
    # regrouping runs must never change what is being measured.
    group: str | None = None

    @property
    def path(self) -> Path:
        return self.run_dir / self.name


class Config(ConfigBase):
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    loss: LossConfig = LossConfig()
    coupling: CouplingConfig = CouplingConfig()
    detector: DetectorConfig = DetectorConfig()
    translator: TranslatorConfig = StubTranslatorConfig()
    metrics: MetricsConfig = MetricsConfig()
    export: ExportConfig = ExportConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def load(
        cls,
        path: Path | str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build a config from an optional YAML file plus optional nested overrides.

        Overrides win over the file, which wins over the declared defaults.
        """
        merged: dict[str, Any] = {}
        if path is not None:
            loaded = yaml.safe_load(Path(path).read_text()) or {}
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"{path}: expected a mapping at the top level")
            merged = deep_merge(merged, loaded)
        if overrides:
            merged = deep_merge(merged, overrides)
        try:
            return cls.model_validate(merged)
        except ValidationError as exc:
            raise as_config_error(exc) from exc

    def to_dict(self) -> dict[str, Any]:
        """The resolved config as YAML-safe primitives: Path -> str, enum -> str."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """Short digest identifying the *experiment*, stored in checkpoints to catch drift.

        `runtime` is excluded wholly: device, run name and W&B flags do not change what is
        measured, so two invocations of one experiment must agree here. Diverges from
        ``../Clean-SeAFusion/src/seafusion/engine/fusion_trainer.py:60``, which hashed
        everything and therefore warned on a renamed run.
        """
        payload = self.model_dump(mode="json", exclude={"runtime"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def snapshot(self, run_dir: Path | None = None) -> Path:
        """Write the fully resolved config into the run directory, for reproducibility.

        Includes `runtime`, unlike the hash -- provenance wants to know which machine.
        """
        target = Path(run_dir) if run_dir is not None else self.runtime.path
        target.mkdir(parents=True, exist_ok=True)
        destination = target / CONFIG_FILENAME
        destination.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return destination
