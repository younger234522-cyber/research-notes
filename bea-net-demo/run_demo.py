"""Small CPU demo using the unchanged official BEA-Net backbone and demo preprocessing."""
import argparse
import csv
import hashlib
import json
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_COMMIT = '610823cc656c00661e5760031aa26ac13929c0d9'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom


def metrics(pred, target):
    pred, target = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError('Prediction and target must have the same shape')
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    denominator = 2 * tp + fp + fn
    return {'dice': 2 * tp / denominator if denominator else 1.0,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'false_positive_pixels': fp, 'false_negative_pixels': fn}


def load_weights(path, model):
    # Permit only known NumPy scalar metadata in this author's legacy checkpoint.
    # Do not switch to unrestricted pickle loading.
    safe = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
            np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    adapted = {}
    for name, value in state.items():
        name = name.removeprefix('module.').removeprefix('U_ResTran3D.backbone.')
        if name in adapted:
            raise ValueError('Duplicate checkpoint key: ' + name)
        adapted[name] = value
    model.load_state_dict(adapted, strict=True)
    return checkpoint.get('epoch'), len(adapted)


def draw_comparison(image_path, gt, pred, score, out):
    rgb = np.array(Image.open(image_path).convert('RGB'))
    rgb = zoom(rgb, (256 / rgb.shape[0], 256 / rgb.shape[1], 1), order=3)
    errors = np.zeros((256, 256, 3), dtype=np.uint8)
    errors[pred & ~gt] = [220, 65, 50]
    errors[~pred & gt] = [45, 105, 220]
    errors[pred & gt] = [190, 190, 190]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.3), layout='constrained')
    for ax, data, title in zip(axes, [rgb, gt, pred, errors],
                              ['Public dermoscopy image', 'Reference mask',
                               'BEA-Net prediction', 'Red: extra | Blue: missed']):
        ax.imshow(data, cmap='gray', vmin=0, vmax=1 if data.ndim == 2 else 255)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.suptitle(f'{image_path.stem} | Dice {score["dice"]:.4f} | Demo only, not an independent benchmark', fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=ROOT / 'vendor' / 'BEA-Net')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error('Checkpoint file not found: ' + str(args.checkpoint))
    if args.limit < 1 or args.threads < 1:
        parser.error('limit and threads must be positive')
    # Lightweight correctness checks for the metric implementation.
    assert metrics(np.ones((2, 2)), np.ones((2, 2)))['dice'] == 1
    assert metrics(np.zeros((2, 2)), np.ones((2, 2)))['dice'] == 0
    assert metrics(np.zeros((2, 2)), np.zeros((2, 2)))['dice'] == 1
    assert metrics([1, 1, 0], [1, 0, 1])['dice'] == 0.5
    repo = args.repo.resolve()
    required = [repo / 'BEA_package/BEA/network_architecture/bsg.py',
                repo / 'BEA_package/BEA/demo/dataset.py', repo / 'nnUNet']
    if not all(p.exists() for p in required):
        parser.error('Official BEA-Net checkout is missing; see README or supply --repo')
    sys.path[:0] = [str(repo / 'BEA_package'), str(repo / 'nnUNet'),
                   str(repo / 'BEA_package/BEA/demo')]
    from BEA.network_architecture.bsg import UNet
    from dataset import preprocess
    try:
        repo_commit = subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL).strip()
        repo_dirty = bool(subprocess.check_output(
            ['git', '-C', str(repo), 'status', '--porcelain'], text=True,
            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        repo_commit, repo_dirty = None, None
    if repo_commit != EXPECTED_COMMIT or repo_dirty is not False:
        print('WARNING: upstream revision/clean state not verified; see README.', flush=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; rerun with --device cpu')
    torch.set_num_threads(args.threads)
    model = UNet(in_channels=3, num_classes=2, base_num_features=32,
                 n_layer=5, convolutional_upsampling=True,
                 norm_cfg='IN', activation_cfg='LeakyReLU')
    epoch, loaded_entries = load_weights(args.checkpoint, model)
    print(f'Strict checkpoint match passed: {loaded_entries} entries; epoch={epoch}', flush=True)
    model = model.to(args.device).eval()
    data_dir = repo / 'BEA_package' / 'BEA' / 'demo' / 'images'
    images = sorted(data_dir.glob('ISIC_*.jpg'))[:args.limit]
    if not images:
        raise RuntimeError('No official example images found')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images:
        label_path = path.with_name(path.stem + '_segmentation.png')
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        # Preserve the author's demo preprocessing (RGB, cubic resize,
        # per-channel z-score; demo's linear mask resize and integer cast).
        array = preprocess(path).astype(np.float32)
        target = preprocess(label_path, is_mask=True).astype(np.int64).astype(bool)
        image_tensor = torch.from_numpy(array[None]).to(args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            logits, _, _ = model(image_tensor)
            if logits.shape != (1, 2, 256, 256) or not torch.isfinite(logits).all():
                raise RuntimeError('Unexpected model output')
            # Argmax of logits equals argmax of softmax in the official demo.
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(bool)
        elapsed = time.perf_counter() - start
        score = metrics(pred, target)
        row = {'image_id': path.stem, **score, 'inference_seconds': round(elapsed, 3)}
        rows.append(row)
        Image.fromarray(pred.astype(np.uint8) * 255).save(args.output / f'{path.stem}_prediction.png')
        draw_comparison(path, target, pred, score, args.output / f'{path.stem}_comparison.png')
        print(json.dumps(row), flush=True)
    with (args.output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.checkpoint.open('rb') as checkpoint_stream:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_stream, 'sha256').hexdigest()
    report = {
        'status': 'pretrained_demo_completed', 'device': args.device,
        'python': sys.version, 'torch': torch.__version__,
        'checkpoint_filename': args.checkpoint.name,
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_epoch': epoch, 'strict_matched_entries': loaded_entries,
        'trainable_parameter_count': sum(p.numel() for p in model.parameters()),
        'repo_commit': repo_commit, 'repo_dirty': repo_dirty,
        'expected_repo_commit': EXPECTED_COMMIT,
        'mean_dice': float(np.mean([r['dice'] for r in rows])), 'cases': rows,
        'protocol': 'Unchanged official UNet backbone and official demo preprocessing; '
                    'single 256x256 pass, eval mode, no TTA, no training. '
                    'The wrapper only squeezes/unsqueezes a singleton depth axis; '
                    'its parameter-name prefix is removed for strict backbone loading.',
        'limitation': 'These four repository examples are NOT an independent test set. '
                      'Training overlap is unknown. Metrics at 256x256 follow demo mask preprocessing '
                      '(linear resize then integer cast), not a full paper benchmark reproduction.'
    }
    (args.output / 'run_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'DONE: {len(rows)} images, demo mean Dice={report["mean_dice"]:.4f}', flush=True)


if __name__ == '__main__':
    main()

