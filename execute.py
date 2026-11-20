#!/usr/bin/env python3
"""Run SAT-HMR -> SAM 2 -> GGDO -> PowerPaint for one input image.

Examples:
    python3 execute.py /path/to/photo.jpg
    python3 execute.py                     # uses data/football.png
    python3 execute.py --check             # inspect installed prerequisites

Model weights and a compatible CUDA environment must be installed first.
Multi-view synthesis, NeuS and PAPO are not implemented by the local pipeline.
Additional options from `python -m decon.cli --help` may be passed through.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / 'data/football.png'  # Change this to use another default photo.
OFFICIAL_ROOT = ROOT / 'vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou'


def environment_python():
    for name in ('.venv-gpu', '.venv'):
        candidate = ROOT / name / 'bin/python'
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def absolute(value):
    return Path(value).expanduser().resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', nargs='?', default=DEFAULT_IMAGE,
                        help='Input image path (default: data/football.png)')
    parser.add_argument('--output', type=absolute, help='New output directory')
    parser.add_argument('--python', default=environment_python(),
                        help='Python interpreter with CUDA/model dependencies (auto-detects project venv)')
    parser.add_argument('--official-root', type=absolute, default=OFFICIAL_ROOT)
    parser.add_argument('--sam-checkpoint', type=absolute)
    parser.add_argument('--sam-config', help='SAM config matching the checkpoint')
    parser.add_argument('--powerpaint-checkpoint', type=absolute,
                        help='PowerPaint ppt-v2 directory')
    parser.add_argument('--allow-model-downloads', action='store_true',
                        help='Allow PowerPaint to download its base models')
    parser.add_argument('--check', action='store_true', help='Report setup without running inference')
    args, extra = parser.parse_known_args(argv)
    if args.check:
        return subprocess.run([args.python, str(ROOT / 'scripts/check_official_setup.py'),
                               '--official-root', str(args.official_root)], cwd=ROOT).returncode

    image = absolute(args.image)
    if not image.is_file():
        parser.error(f'Input image does not exist: {image}')
    output = args.output or ROOT / 'outputs' / f'{image.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if output.exists():
        parser.error(f'Output directory already exists; choose a new path: {output}')

    # The bundled checkpoint is base-plus, which requires its own matching config.
    if args.sam_checkpoint is None:
        candidates = [
            (ROOT / 'checkpoints/sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
            (ROOT / 'checkpoints/sam2.1_hiera_large.pt', 'configs/sam2.1/sam2.1_hiera_l.yaml'),
            (args.official_root / 'Img_Segmentation/checkpoints/sam2.1_hiera_large.pt',
             'configs/sam2.1/sam2.1_hiera_l.yaml'),
        ]
        args.sam_checkpoint, config = next(((p, c) for p, c in candidates if p.is_file()), candidates[0])
        args.sam_config = args.sam_config or config
    elif not args.sam_config:
        parser.error('Supply --sam-config when specifying a custom --sam-checkpoint')
    powerpaint = args.powerpaint_checkpoint
    if powerpaint is None:
        candidates = [ROOT / 'checkpoints/ppt-v2',
                      args.official_root / 'Img_Inpainting/checkpoints/ppt-v2']
        powerpaint = next((p for p in candidates if p.is_dir()), candidates[0])

    command = [args.python, '-m', 'decon.cli', str(image),
               '--official-root', str(args.official_root), '--output', str(output),
               '--sam-checkpoint', str(args.sam_checkpoint), '--sam-config', args.sam_config,
               '--powerpaint-checkpoint', str(powerpaint)]
    if args.allow_model_downloads:
        command.append('--allow-model-downloads')
    command.extend(extra)
    print('Running body estimation -> segmentation -> geometry fitting -> completion.', flush=True)
    print(shlex.join(command), flush=True)
    environment = os.environ.copy()
    runtime = ROOT / '.venv-gpu'
    if runtime.is_symlink():
        # Keep downloaded base models beside the externally stored environment.
        environment.setdefault('HF_HOME', str(runtime.resolve().parent / 'huggingface'))
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode:
        print('\nPipeline did not complete. See the error above. Check prerequisites with:\n'
              f'  {shlex.join([sys.executable, str(ROOT / "execute.py"), "--python", args.python, "--check"])}\n'
              'See README.md for checkpoint and CUDA setup instructions.', file=sys.stderr)
    else:
        print(f'Outputs: {output}\nEach person folder contains completed.png, masks, overlays and an optimized mesh.')
    return result.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, KeyboardInterrupt) as error:
        print(f'Unable to finish: {error}', file=sys.stderr)
        raise SystemExit(1)
