"""Prepare locally downloaded SAT-HMR and SMPL assets for official DECON."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SMPL_RENAMES = {
    "basicModel_f_lbs_10_207_0_v1.0.0.pkl": "SMPL_FEMALE.pkl",
    "basicModel_m_lbs_10_207_0_v1.0.0.pkl": "SMPL_MALE.pkl",
    "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl": "SMPL_NEUTRAL.pkl",
}


def prepare(source: Path, weights_root: Path) -> list[str]:
    smpl_destination = weights_root / "smpl_data" / "smpl"
    sat_destination = weights_root / "sat_hmr"
    smpl_destination.mkdir(parents=True, exist_ok=True)
    sat_destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source_name, target_name in SMPL_RENAMES.items():
        source_file = source / source_name
        if source_file.is_file():
            target_file = smpl_destination / target_name
            if target_file.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {target_file}")
            shutil.copy2(source_file, target_file)
            copied.append(target_name)

    for checkpoint in sorted(source.glob("*.pth")):
        target_file = sat_destination / checkpoint.name
        if not target_file.exists():
            shutil.copy2(checkpoint, target_file)
            copied.append(checkpoint.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing downloaded SMPL/SAT-HMR files")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("../DECON_official/Multi-View_Synthesis/Geo_Gui_Hm_Decou/Human_SMPL_Estimation/weights"),
    )
    args = parser.parse_args()
    copied = prepare(args.source, args.destination)
    print(f"Copied {len(copied)} file(s) to {args.destination}")
    for name in copied:
        print(f"  {name}")
    missing = [name for name in SMPL_RENAMES.values() if not (args.destination / "smpl_data" / "smpl" / name).is_file()]
    if missing:
        print("Missing licensed SMPL files:")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
