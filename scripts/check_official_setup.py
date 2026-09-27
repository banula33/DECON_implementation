import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decon.preflight import format_preflight, inspect_official_setup


if __name__ == "__main__":
    official_root = Path(__file__).resolve().parents[2] / "DECON_official" / "Multi-View_Synthesis" / "Geo_Gui_Hm_Decou"
    print(format_preflight(inspect_official_setup(official_root)))
