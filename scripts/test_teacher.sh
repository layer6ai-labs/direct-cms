export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CUDA_VISIBLE_DEVICES=0 python test_teacher.py \
    --pretrained_teacher_model=${MODEL_NAME}  \
    --pretrained_vae_model_name_or_path=${VAE_PATH} \
    --output_dir="results/teacher-ddim" \
    --weight_dtype="fp16" \
    --enable_xformers_memory_efficient_attention \
    --num_original_inference_steps=100 \
    --num_images_per_prompt=2 \
    --scheduler="ddim" \
    --guidance_scale=9.0 \
    --train_data_dir="data/laion_aes/preprocessed_11k/train" \
    --dgm_metrics="fd" \
    --dgm_metrics_batch_size=8 \
    --seed=24 \
