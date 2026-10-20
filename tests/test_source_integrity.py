from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "run_robust_ga_v2_300.py": "5573751893c2d13e741a65f149e95a23faaa1ab0a7e9ac8939550fc66fc488cd",
    "run_robust_ga_v2m.py": "3fe75f2078ef278be32989a3a8e8fa8014c0d3d35b5b544bd31e8c3d307b729f",
    "run_robust_ga_v3.py": "ef547aaa9871936a34818285fbfac4e26e42e0044117bc0318b79c7088dcfbc6",
    "run_robust_ga_v3m.py": "419ad99d4996cb52b3b5465057c21ae6a714ebfc57966ae15990359a46a62504",
}

def test_frozen_ga_sources_match_original_hashes():
    archive = ROOT / "provenance" / "hash_locked_search_sources.zip"
    with zipfile.ZipFile(archive) as z:
        for name, expected in EXPECTED.items():
            b = z.read("scripts/optimizer_tail_v2/" + name)
            assert hashlib.sha256(b).hexdigest() == expected
