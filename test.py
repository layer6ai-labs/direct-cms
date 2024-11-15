import gc
import os
import argparse
import logging
from logging import getLogger
from contextlib import nullcontext
from pathlib import Path
from packaging import version
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from scheduler_lcm import LCMScheduler
from instance_metrics import InstanceMetrics
from dgm_metrics import DGMMetrics

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = getLogger(__name__)


def log_testing(vae, args, writer, device, weight_dtype):
    logger.info("Running testing... ")

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_teacher_model,
        vae=vae,
        scheduler=LCMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler"),
        revision=args.revision,
        torch_dtype=weight_dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.scheduler.config.num_original_inference_steps = args.num_original_inference_steps

    checkpoints = os.listdir(args.output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    latest = checkpoints[-1]

    to_load = Path(args.output_dir, latest)

    pipeline.load_lora_weights(to_load)
    pipeline.fuse_lora(lora_scale=args.lora_scale)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    pipeline.unet.eval()
    
    instance_metrics = InstanceMetrics(device)
    if os.path.exists(Path(args.train_data_dir, "latents.pt")):
        real_reps_dir = Path(args.train_data_dir) / "real_reps"
    else:
        real_reps_dir = Path(args.train_data_dir).parent / "real_reps"
    dgm_metrics = DGMMetrics(device, args.dgm_metrics, args.output_dir, str(real_reps_dir), args.dgm_metrics_batch_size)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    validation_prompts = [
        "cute sundar pichai character",
        "robotic cat with wings",
        "a photo of yoda",
        "a cute creature with blue eyes",
    ]

    with open("data/train_captions.txt", "r") as f:
        train_prompts = f.read().splitlines()

    image_logs = []
    for _, prompt in tqdm(enumerate(validation_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(device.type, dtype=weight_dtype)

        with autocast_ctx:
            images = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})

    init_noise = None
    if os.path.exists("data/init_noise.pt"):
        logger.info("Using custom noise samples for sampling... ")
        init_noise = torch.load("data/init_noise.pt")
        assert len(init_noise) == len(train_prompts)

    train_gen_images_dir = Path(args.output_dir, "train_gen_images", f"{args.num_inference_steps}_steps")
    train_gen_images_dir.mkdir(parents=True)
    latents, clip_scores, aesthetic_scores = [], [], []
    for i, prompt in tqdm(enumerate(train_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(device.type, dtype=weight_dtype)

        with autocast_ctx:
            latent = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
                latents=init_noise[i].unsqueeze(0) if init_noise is not None else None,
                guidance_scale=args.guidance_scale,
                output_type="latent",
            ).images
            latents.append(latent[0].cpu())

            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

            if needs_upcasting:
                pipeline.upcast_vae()
                latent = latent.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
            elif latent.dtype != pipeline.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    pipeline.vae = pipeline.vae.to(latent.dtype)

            # unscale/denormalize the latent
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(pipeline.vae.config, "latents_mean") and pipeline.vae.config.latents_mean is not None
            has_latents_std = hasattr(pipeline.vae.config, "latents_std") and pipeline.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latent.device, latent.dtype)
                )
                latents_std = (
                    torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(latent.device, latent.dtype)
                )
                latent = latent * latents_std / pipeline.vae.config.scaling_factor + latents_mean
            else:
                latent = latent / pipeline.vae.config.scaling_factor

            image = pipeline.vae.decode(latent, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                pipeline.vae.to(dtype=torch.float16)

            # apply watermark if available
            if pipeline.watermark is not None:
                image = pipeline.watermark.apply_watermark(image)

            image = pipeline.image_processor.postprocess(image, output_type="pil")[0]

            # Offload all models
            pipeline.maybe_free_model_hooks()

        clip_score, aesthetic_score = instance_metrics.compute_instance_metrics(image, prompt)
        clip_scores.append(clip_score)
        aesthetic_scores.append(aesthetic_score)
        image.save(train_gen_images_dir / f"{i}.jpg")

    avg_clip_score = sum(clip_scores) / len(clip_scores)
    avg_aesthetic_score = sum(aesthetic_scores) / len(aesthetic_scores)

    latents = torch.stack(latents).float()
    torch.save(latents, train_gen_images_dir / "latents.pt")
    if os.path.exists(Path(args.train_data_dir, "latents.pt")):
        latents_train = torch.load(Path(args.train_data_dir, "latents.pt"))
        latent_mse = torch.nn.functional.mse_loss(latents, latents_train)
    else:
        logger.info(f"The train_data_dir {args.train_data_dir} does not have a latents.pt. Cannot compute latent MSE.")
        latent_mse = None

    logger.info("Computing DGM eval metrics... ")
    train_real_images_dir = args.train_data_dir
    dgm_inception_scores = dgm_metrics.compute_dgm_metrics([train_real_images_dir, str(train_gen_images_dir)], "inception")["run00"]
    logger.info("Computed Inception DGM eval metrics... ")
    dgm_dinov2_scores = dgm_metrics.compute_dgm_metrics([train_real_images_dir, str(train_gen_images_dir)], "dinov2")["run00"]
    logger.info("Computed DINOv2 DGM eval metrics... ")

    tag_suffix = f"[{args.num_inference_steps} steps]"
    for log in image_logs:
        images = log["images"]
        validation_prompt = f"{log['validation_prompt']} {tag_suffix}"
        formatted_images = []
        for image in images:
            formatted_images.append(np.asarray(image))

        formatted_images = np.stack(formatted_images)

        writer.add_images(validation_prompt, formatted_images, 0, dataformats="NHWC")

    writer.add_scalar(f"clip score {tag_suffix}", avg_clip_score, 0)
    writer.add_scalar(f"aesthetic score {tag_suffix}", avg_aesthetic_score, 0)

    if latent_mse is not None:
        writer.add_scalar(f"latent mse {tag_suffix}", latent_mse, 0)

    for k, v in dgm_inception_scores.items():
        writer.add_scalar(f"inception {k} score {tag_suffix}", v, 0)

    for k, v in dgm_dinov2_scores.items():
        writer.add_scalar(f"dinov2 {k} score {tag_suffix}", v, 0)

    writer.close()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def parse_args():
    parser = argparse.ArgumentParser(description="Testing script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    # ----Sampling Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The directory where the model predictions and checkpoints will be loaded from and written to.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible sampling.")
    parser.add_argument("--num_original_inference_steps", type=int, default=100, help="Default number of inference steps")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps during validation")
    parser.add_argument("--num_images_per_prompt", type=int, default=4, help="Number of images to generate per prompt")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Imagen classifier-free guidance scale")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Controls how much to influence the outputs with the LoRA parameters")
    # ----Metric Arguments----
    parser.add_argument("--train_data_dir", type=str, help="Path to training images for computing DGM eval metrics")
    parser.add_argument("--dgm_metrics", type=str, nargs="+", default=["fd", "fd-infinity", "kd", "prdc",
                                                                       "is", "authpct", "ct", "ct_test", "ct_modified", 
                                                                       "fls", "fls_overfit", "vendi", "sw_approx"],
                        help="metrics to compute")
    parser.add_argument("--dgm_metrics_batch_size", type=int, default=8, help="Batch size to compute DGM metrics")
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs/text2image-fine-tune",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    # ----Optimizations----
    parser.add_argument("--weight_dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    assert os.path.exists(args.output_dir)

    # Load VAE from SDXL checkpoint (or more stable VAE)
    vae_path = (
        args.pretrained_teacher_model
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.teacher_revision,
    )
    vae.requires_grad_(False)

    # Handle mixed precision and device placement
    weight_dtype = torch.float32
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.pretrained_vae_model_name_or_path is None:
        vae.to(device, dtype=torch.float32)
    else:
        vae.to(device, dtype=weight_dtype)

    # Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    logging_dir = Path(args.output_dir, args.logging_dir)
    writer = SummaryWriter(log_dir=logging_dir)

    log_testing(vae, args, writer, device, weight_dtype)


if __name__ == "__main__":
    args = parse_args()
    main(args)
