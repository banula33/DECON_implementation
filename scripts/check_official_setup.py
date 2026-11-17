"""Inspect the decoupling dependencies and local model assets."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decon.preflight import format_preflight, inspect_official_setup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path,
                        default=Path('vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou'))
    args = parser.parse_args()
    print(format_preflight(inspect_official_setup(args.official_root)))
