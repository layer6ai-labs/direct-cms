export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --pretrained_teacher_model=${MODEL_NAME}  \
    --pretrained_vae_model_name_or_path=${VAE_PATH} \
    --output_dir="results/cm-ddim100-w9-lr1e4-bs16-250steps" \
    --weight_dtype="fp16" \
    --enable_xformers_memory_efficient_attention \
    --num_original_inference_steps=100 \
    --num_inference_steps=1 \
    --num_images_per_prompt=2 \
    --guidance_scale=0.0 \
    --lora_scale=1.0 \
    --train_data_dir="results/teacher-ddim/train_gen_images/100_steps/9.0_gscale" \
    --dgm_metrics="fd" \
    --dgm_metrics_batch_size=8 \
    --seed=24 \
