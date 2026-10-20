import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def test_python_scripts_syntax():
    originals = list((ROOT/'code').rglob('*.py'))
    assert len(originals)==66
    for p in originals:ast.parse(p.read_text(encoding='utf-8'), filename=str(p))

def test_shell_and_slurm_syntax():
    shell=[p for p in ROOT.rglob('*') if p.suffix in ('.sh','.sbatch')]
    assert len(shell)==64
    for p in shell:
        check=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
        assert check.returncode==0, f'{p}: {check.stderr}'


def test_recovered_external_openmc_python_syntax():
    external = list((ROOT / 'code' / 'annealing').glob('*.py'))
    assert len(external) == 2
    for p in external:
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
