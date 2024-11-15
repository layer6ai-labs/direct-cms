import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import torch
from dgm_eval.dataloaders import get_dataloader
from dgm_eval.metrics import *
from dgm_eval.models import load_encoder, InceptionEncoder, MODELS
from dgm_eval.representations import get_representations, load_reps_from_path, save_outputs


def get_device_and_num_workers(device, num_workers):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = num_workers

    return device, num_workers


def get_dataloader_from_path(path, model_transform, num_workers, args, sample_w_replacement=False):
    print(f'Getting DataLoader for path: {path}\n', file=sys.stderr)

    dataloader = get_dataloader(path, args.nsample, args.batch_size, num_workers, seed=args.seed,
                                sample_w_replacement=sample_w_replacement,
                                transform=lambda x: model_transform(x))

    return dataloader


def compute_representations(DL, model, device, args, is_real_reps=False):
    """
    Load representations from disk if path exists,
    else compute image representations using the specified encoder

    Returns:
        repsi: float32 (Nimage, ndim)
    """

    if is_real_reps and args.load_real_reps:
        print(f'Loading saved real representations from: {args.real_reps_dir}\n', file=sys.stderr)
        repsi = load_reps_from_path(args.real_reps_dir, args.model, None, DL)
        if repsi is not None: 
            return repsi

        print(f'No saved representations found: {args.real_reps_dir}\n', file=sys.stderr)

    print('Calculating Representations\n', file=sys.stderr)
    repsi = get_representations(model, DL, device, normalized=False)
    if is_real_reps and args.save_real_reps:
        print(f'Saving real representations to {args.real_reps_dir}\n', file=sys.stderr)
        save_outputs(args.real_reps_dir, repsi, args.model, None, DL)
    return repsi


def compute_scores(args, reps, test_reps, labels=None):

    scores={}
    vendi_scores = None

    if 'fd' in args.metrics:
        print("Computing FD \n", file=sys.stderr)
        scores['fd'] = compute_FD_with_reps(*reps)

    if 'fd_eff' in args.metrics:
        print("Computing Efficient FD \n", file=sys.stderr)
        scores['fd_eff'] = compute_efficient_FD_with_reps(*reps)

    if 'fd-infinity' in args.metrics:
        print("Computing fd-infinity \n", file=sys.stderr)
        scores['fd_infinity_value'] = compute_FD_infinity(*reps)

    if 'kd' in args.metrics:
        print("Computing KD \n", file=sys.stderr)
        mmd_values = compute_mmd(*reps)
        scores['kd_value'] = mmd_values.mean()
        scores['kd_variance'] = mmd_values.std()

    if 'prdc' in args.metrics:
        print("Computing precision, recall, density, and coverage \n", file=sys.stderr)
        reduced_n = min(args.reduced_n, reps[0].shape[0], reps[1].shape[0])
        inds0 = np.random.choice(reps[0].shape[0], reduced_n, replace=False)

        inds1 = np.arange(reps[1].shape[0])
        if 'realism' not in args.metrics:
            # Realism is returned for each sample, so do not shuffle if this metric is desired.
            # Else filenames and realism scores will not align
            inds1 = np.random.choice(inds1, min(inds1.shape[0], reduced_n), replace=False)

        prdc_dict = compute_prdc(
            reps[0][inds0], 
            reps[1][inds1], 
            nearest_k=args.nearest_k,
            realism=True if 'realism' in args.metrics else False)
        scores = dict(scores, **prdc_dict)

    if 'vendi' in args.metrics:
        print("Calculating diversity score", file=sys.stderr)
        # scores['vendi'] = compute_vendi_score(reps[1])
        vendi_scores = compute_per_class_vendi_scores(reps[1], labels)
        scores['mean vendi per class'] = vendi_scores.mean()

    if 'authpct' in args.metrics:
        print("Computing authpct \n", file=sys.stderr)
        scores['authpct'] = compute_authpct(*reps)

    if 'sw_approx' in args.metrics:
        print('Aprroximating Sliced W2.', file=sys.stderr)
        scores['sw_approx'] = sw_approx(*reps)

    if 'ct' in args.metrics:
        print("Computing ct score \n", file=sys.stderr)
        scores['ct'] = compute_CTscore(reps[0], test_reps, reps[1])

    if 'ct_test' in args.metrics:
        print("Computing ct score, modified to identify mode collapse only \n", file=sys.stderr)
        scores['ct_test'] = compute_CTscore_mode(reps[0], test_reps, reps[1])

    if 'ct_modified' in args.metrics:
        print("Computing ct score, modified to identify memorization only \n", file=sys.stderr)
        scores['ct_modified'] = compute_CTscore_mem(reps[0], test_reps, reps[1])

    if 'fls' in args.metrics or 'fls_overfit' in args.metrics:
        train_reps, gen_reps = reps[0], reps[1]
        reduced_n = min(args.reduced_n, train_reps.shape[0]//2, test_reps.shape[0], gen_reps.shape[0])

        test_reps = test_reps[np.random.choice(test_reps.shape[0], reduced_n, replace=False)]
        gen_reps = gen_reps[np.random.choice(gen_reps.shape[0], reduced_n, replace=False)]

        print("Computing fls \n", file=sys.stderr)
        # fls must be after ot, as it changes train_reps
        train_reps = train_reps[np.random.choice(train_reps.shape[0], 2*reduced_n, replace=False)]
        train_reps, baseline_reps = train_reps[:reduced_n], train_reps[reduced_n:]

        if 'fls' in args.metrics:
            scores['fls'] = compute_fls(train_reps, baseline_reps, test_reps, gen_reps)
        if 'fls_overfit' in args.metrics:
            scores['fls_overfit'] = compute_fls_overfit(train_reps, baseline_reps, test_reps, gen_reps)

    for key, value in scores.items():
        if key=='realism': continue
        print(f'{key}: {value:.5f}\n')

    return scores, vendi_scores


def save_score(scores, output_dir, model, path, ckpt, nsample, is_only=False):

    ckpt_str = ''
    if ckpt is not None:
        ckpt_str = f'_ckpt-{os.path.splitext(os.path.basename(ckpt))[0]}'

    if is_only:
        out_str = f"Inception_score_{'-'.join([os.path.basename(p) for p in path])}{ckpt_str}_nimage-{nsample}.txt"
    else:
        out_str = f"fd_{model}_{'-'.join([os.path.basename(p) for p in path])}{ckpt_str}_nimage-{nsample}.txt"

    out_path = os.path.join(output_dir, out_str)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        for key, value in scores.items():
            if key=='realism': continue
            f.write(f"{key}: {value} \n")


def save_scores(scores, args, is_only=False, vendi_scores={}):

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_params = vars(args)
    run_params['reference_dataset'] = run_params['path'][0]
    run_params['test_datasets'] = run_params['path'][1:]

    ckpt_str = ''
    print(scores, file=sys.stderr)

    if is_only: 
        out_str = f'Inception_scores_nimage-{args.nsample}'
    else:
        out_str = f"{args.model}{ckpt_str}_scores_nimage-{args.nsample}"
    out_path = os.path.join(args.output_dir, out_str)

    np.savez(f'{out_path}.npz', scores=scores, run_params=run_params)

    if vendi_scores is not None and len(vendi_scores)>0:
        df = pd.DataFrame.from_dict(data=vendi_scores)
        out_str = f"{args.model}{ckpt_str}_vendi_scores_nimage-{args.nsample}"
        out_path = os.path.join(args.output_dir, out_str)
        print(f'saving vendi score to {out_path}.csv')
        df.to_csv(f'{out_path}.csv')


def get_inception_scores(args, device, num_workers):
    # The inceptionV3 with logit output is only used for calculate inception score
    print(f'Computing Inception score with model = inception, ckpt=None, and dims=1008.', file=sys.stderr)
    print('Loading Model', file=sys.stderr)

    IS_scores = {}
    model_IS = load_encoder('inception', device, ckpt=None,
                        dims=1008, arch=None,
                        pretrained_weights=None,
                        train_dataset=None,
                        clean_resize=args.clean_resize,
                        depth=args.depth)
    
    for i, path in enumerate(args.path[1:]):
        print(f'Getting DataLoader for path: {path}\n', file=sys.stderr)
        dataloaderi = get_dataloader_from_path(args.path[i], model_IS.transform, num_workers, args)
        print(f'Computing inception score for {path}\n', file=sys.stderr)
        IS_score_i = compute_inception_score(model_IS, DataLoader=dataloaderi, splits=args.splits, device=device)
        IS_scores[f'run{i:02d}'] = IS_score_i
        print(IS_score_i)
    save_scores(IS_scores, args, is_only=True)
    if len(args.metrics) == 1: sys.exit(0)

    return IS_scores


@dataclass
class Args:
    model: str = "dinov2"
    train_dataset: str = "imagenet"
    batch_size: int = 50
    num_workers: Optional[int] = None
    device: str = None
    nearest_k: int = 5
    reduced_n: int = 10000
    nsample: int = 50000
    path: list[str] = field(default_factory=lambda: [""])
    test_path: str = None
    metrics: list[str] = field(default_factory=lambda: ["fd", "fd-infinity", "kd", "prdc", 
                                                         "is", "authpct", "ct", "ct_test", "ct_modified", 
                                                         "fls", "fls_overfit", "vendi", "sw_approx"])
    checkpoint: str = None
    arch: str = None
    splits: int = 10
    output_dir: str = "results/"
    real_reps_dir: str = "data/real_reps/"
    save_real_reps: bool = True
    load_real_reps: bool = True
    seed: int = 13579
    clean_resize: bool = False
    depth: int = 0


class DGMMetrics():
    def __init__(self, device, metrics, output_dir, real_reps_dir, batch_size=50):
        self.args = Args()
        self.args.device = device
        self.args.metrics = metrics
        self.args.output_dir = output_dir
        self.args.real_reps_dir = real_reps_dir
        self.args.batch_size = batch_size

        self.device, self.num_workers = get_device_and_num_workers(self.args.device, self.args.num_workers)

    @torch.inference_mode()    
    def compute_dgm_metrics(self, path, model="dinov2"):
        self.args.path = path
        self.args.model = model

        IS_scores = None
        if 'is' in self.args.metrics and self.args.model == 'inception':
            # Does not require a reference dataset, so compute first.
            IS_scores = get_inception_scores(self.args, self.device, self.num_workers)

        print('Loading Model', file=sys.stderr)
        # Get train representations
        model = load_encoder(self.args.model, self.device, ckpt=None, arch=None,
                            clean_resize=self.args.clean_resize,
                            sinception=True if self.args.model=='sinception' else False,
                            depth=self.args.depth,
                            )
        dataloader_real = get_dataloader_from_path(self.args.path[0], model.transform, self.num_workers, self.args)
        reps_real = compute_representations(dataloader_real, model, self.device, self.args, is_real_reps=True)

        # Get test representations
        repsi_test = None
        if self.args.test_path is not None:
            dataloader_test = get_dataloader_from_path(self.args.test_path, model.transform, self.num_workers, self.args)

            repsi_test = compute_representations(dataloader_test, model, self.device, self.args, is_real_reps=True)

        # Loop over all generated paths
        all_scores = {}
        vendi_scores = {}
        for i, path in enumerate(self.args.path[1:]):

            dataloaderi = get_dataloader_from_path(path, model.transform, self.num_workers, self.args,
                                                sample_w_replacement=True if ':train' in path else False)
            repsi = compute_representations(dataloaderi, model, self.device, self.args)
            reps = [reps_real, repsi]

            print(f'Computing scores between reference dataset and {path}\n', file=sys.stderr)
            scores_i, vendi_scores_i = compute_scores(self.args, reps, repsi_test, dataloaderi.labels)
            if vendi_scores_i is not None:
                vendi_scores[os.path.basename(path)] = vendi_scores_i

            print('Saving scores\n', file=sys.stderr)
            save_score(
                scores_i, self.args.output_dir, self.args.model, [self.args.path[0], path], None, self.args.nsample,
            )
            if IS_scores is not None:
                scores_i.update(IS_scores[f'run{i:02d}'])
            all_scores[f'run{i:02d}'] = scores_i

        # save scores from all generated paths
        save_scores(all_scores, self.args, vendi_scores=vendi_scores)
        return all_scores