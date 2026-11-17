"""Convert trusted, licensed SMPL pickles from Chumpy to NumPy storage.

Run this as a separate process. The legacy compatibility aliases are only
needed during conversion, never when loading the resulting runtime models.
The source files are preserved and existing destinations are not overwritten.
"""

import argparse
import inspect
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory of SMPL_*.pkl files")
    parser.add_argument("destination", type=Path, help="New output directory")
    parser.add_argument("--chumpy-source", type=Path,
                        help="Optional extracted Chumpy 0.70 source directory")
    args = parser.parse_args()
    sources = sorted(args.source.glob("SMPL_*.pkl"))
    if not sources:
        parser.error("No SMPL_*.pkl models found")
    if args.destination.exists():
        parser.error("Destination must be new; original models are never overwritten")
    if args.chumpy_source:
        sys.path.insert(0, str(args.chumpy_source.resolve()))

    # Chumpy 0.70 predates Python 3.11 and NumPy 1.24. These aliases are
    # confined to this conversion process, which exits after writing files.
    inspect.getargspec = inspect.getfullargspec
    for name, value in {"bool": bool, "int": int, "float": float,
                        "complex": complex, "object": object,
                        "unicode": str, "str": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    import chumpy

    def convert(value):
        if isinstance(value, chumpy.Ch):
            return np.array(value.r, copy=True)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return value

    args.destination.mkdir(parents=True)
    for source in sources:
        with source.open("rb") as stream:
            original = pickle.load(stream, encoding="latin1")
        converted = convert(original)
        for key, value in original.items():
            if isinstance(value, chumpy.Ch):
                np.testing.assert_array_equal(value.r, converted[key])
        destination = args.destination / source.name
        with destination.open("xb") as stream:
            pickle.dump(converted, stream, protocol=4)
        print(f"Converted {source.name}; Chumpy array values verified", flush=True)


if __name__ == "__main__":
    main()
