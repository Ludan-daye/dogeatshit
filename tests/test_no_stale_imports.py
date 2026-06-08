import pathlib, re


def test_no_experiments_imports():
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    pat = re.compile(r'^\s*(from|import)\s+experiments(\.|\s|$)', re.M)
    offenders = [str(p) for p in src.rglob("*.py")
                 if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"stale 'experiments' imports remain in: {offenders}"
