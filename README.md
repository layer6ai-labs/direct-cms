<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p> 

<div align="center">
<h1>
<b>
Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples
</b>
</h1>

<p align="center">
  <a href='https://arxiv.org/abs/2411.08954'><img src='https://img.shields.io/badge/arXiv-2411.08954-b31b1b.svg' /></a>
</p>
  
<h4>
<b>
<a href="https://www.cs.toronto.edu/~nvouitsis/">Noël Vouitsis</a>, Rasa Hosseinzadeh, <a href="https://www.linkedin.com/in/brendan-ross/">Brendan Leigh Ross</a>, <a href="http://linkedin.com/in/valentin-villecroze">Valentin Villecroze</a>, <a href="https://www.cs.toronto.edu/~satyag/">Satya Krishna Gorti</a>, <a href="http://jescresswell.github.io/">Jesse C. Cresswell</a>, <a href="https://sites.google.com/view/gabriel-loaiza-ganem/">Gabriel Loaiza-Ganem</a>
</b>
</h4>
</div>


## Introduction
This repository contains the official implementation of our NeurIPS 2024 workshop paper <a href='https://arxiv.org/abs/2312.10144'>Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples</a>. We release code for both training and evalution of consistency models (CMs) and direct consistency models (Direct CMs). This code supports distillation of SDXL using LoRA and is based on the consistency distillation implementation from the <a href='https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl.py'>diffusers library</a>.

## Setup

### Environment setup 

As a first step, create the following conda environment:

```bash
conda env create --file cm.yml
conda activate cm
```

And initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment:

```bash
accelerate config default
```

### Data setup

To train our models, we use the 11k subset of [LAION](https://laion.ai/blog/laion-400-open-dataset/) similar to [BK-SDM](https://github.com/Nota-NetsPresso/BK-SDM). You must first download the dataset:

```bash
bash scripts/get_laion_data.sh preprocessed_11k
```

### Evaluation setup

Some of our evaluation metrics require downloading a few additional things. First, to measure the aesthetic score of generated samples, you must download the aesthetic predictor model's weights from <a href='https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth'>here</a> and copy them to `data/aesthetics_mlp_weights`.

For FID and FD-DINO metric evaluation, we use the <a href='https://github.com/layer6ai-labs/dgm-eval/tree/master'>deep generative models evaluation library</a>. To install this library, run the following:
```bash
git clone git@github.com:layer6ai-labs/dgm-eval
cd dgm-eval
pip install -e .
cd ..
```

For the FID, FD-DINO, and ODE solving error metrics, you must first run the teacher model to generate reference samples used in the metric computations. Note that the choice of ODE solver, the number of discretization intervals, and the guidance scale for the teacher should match the corresponding choices used for training the model you wish to evaluate. For example, if you want to evaluate a student model distilled from a teacher model using the DDIM ODE solver with 100 discretization intervals and a classifier-free guidance scale of 9.0, you must first run the teacher model to generate references samples with this same hyperparamter selection. To do this, specify `--scheduler`, `--num_original_inference_steps`, and `--guidance_scale` respectively in `scripts/test_teacher.sh` and then run the following:
```bash
bash scripts/test_teacher.sh
```

To ensure that the teacher and student are being evaluated with the same initial noise, you should additionally pre-generate offline a set of random noise tensors (using `diffusers.utils.randn_tensor`) and save them to `data/init_noise.pt`; you will see this referenced in both `test_teacher.py` and `test.py`.

#### Note on guidance scale
We note that, following the diffusers library implemention of the SDXL pipeline, we use the <a href='https://arxiv.org/abs/2205.11487'>Imagen</a> formulation of classifier-free guidance. The corresponding guidance scale from the original classifier-free guidance formulation is simply 1 less than that in the Imagen formulation (e.g., a guidance scale of 9.0 in the Imagen formulation corresponds to a guidance of 8.0 in the original formulation).

## Training

We provide default training scripts for both consistency models (CMs) and direct consistency models (Direct CMs).

#### CMs

```bash
bash scripts/train_cm.sh
```

#### Direct CMs 

```bash
bash scripts/train_direct_cm.sh
```

## Evaluation

The following shows how to evaluate a trained CM or Direct CM model. You must specify the `--output_dir` of where your LoRA (CM or Direct CM) checkpoints are stored. The `--train_data_dir` should point to the directory where the corresponding reference samples generated from teacher (see Evaluation setup section above) are saved. Also make sure to set `--num_original_inference_steps` to the corresponding number of dicretization intervals used to train the model you are evaluating. Evaluation is performed for the number of student sampling steps specified with `--num_inference_steps`. All evaluations are saved to tensorboard. Run the following:
```bash
bash scripts/test.sh
```


## Citation

If you find this work useful in your research, please cite the following paper:

```
@inproceedings{vouitsis2024inconsistencies,
  title={Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples},
  author={Vouitsis, No{\"e}l and Hosseinzadeh, Rasa and Ross, Brendan Leigh and Villecroze, Valentin and Gorti, Satya Krishna and Cresswell, Jesse C and Loaiza-Ganem, Gabriel},
  booktitle={NeurIPS 2024 Workshop on Attributing Model Behavior at Scale}
  year={2024}
}
```