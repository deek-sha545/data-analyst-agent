"""Sandbox: static safety check + isolated execution."""
import os

import pytest

from analyst.sandbox import check_code, run_code
from eval.reference import DATA_DIR


# ------------------------------------------------------------- static check
@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.system('ls')",
        "import subprocess",
        "from os import path",
        "import socket",
        "__import__('os').system('ls')",
        "eval('1+1')",
        "exec('x=1')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "getattr(object, 'x')",
        "import pandas as pd\npd.read_pickle('x.pkl')",
        "import pandas as pd\npd.read_csv('/etc/passwd')",
        "import pandas as pd\npd.read_csv('../secret.csv')",
        "import pandas as pd\npd.read_csv('https://example.com/x.csv')",
        "import pandas as pd\npd.read_csv('~/.ssh/id_rsa')",
        "def f(:\n  pass",
    ],
)
def test_unsafe_code_rejected(code):
    assert check_code(code), f"should be rejected: {code!r}"


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nimport numpy as np\ndf = pd.read_csv('sales.csv')\nprint(df.head())",
        "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.title('Revenue / units')",
        "x = 'a/b'.split('/')\nprint(f'{x[0]}/{x[1]}')",
        "from collections import Counter\nprint(Counter('aab'))",
        "if __name__ == '__main__':\n    print('...')",
    ],
)
def test_normal_analysis_code_allowed(code):
    assert check_code(code) == []


# ---------------------------------------------------------------- execution
def test_runs_pandas_on_copied_data_file():
    code = "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(len(df) > 100)"
    r = run_code(code, [DATA_DIR / "sales.csv"], timeout=60)
    assert r.ok, r.stderr
    assert r.stdout.strip() == "True"


def test_excel_supported():
    code = "import pandas as pd\nprint(pd.read_excel('hr_employees.xlsx').shape[1] > 3)"
    r = run_code(code, [DATA_DIR / "hr_employees.xlsx"], timeout=60)
    assert r.ok and "True" in r.stdout, r.stderr


def test_error_reported_with_clean_traceback():
    code = "import pandas as pd\ndf = pd.DataFrame({'a': [1]})\nprint(df['missing'])"
    r = run_code(code, timeout=60)
    assert not r.ok
    assert "KeyError" in r.error
    assert 'File "analysis.py", line 3' in r.stderr
    assert "site-packages" not in r.stderr  # library frames are filtered out


def test_timeout_kills_slow_code():
    r = run_code("while True:\n    pass", timeout=2)
    assert not r.ok and r.timed_out
    assert "Timed out" in r.error


def test_no_network():
    code = "import pandas as pd\npd.read_csv('http' + 's:' + '//example.com/x.csv')"
    r = run_code(code, timeout=30)
    assert not r.ok
    assert "network" in r.stderr.lower() or "URLError" in r.stderr or "PermissionError" in r.stderr


def test_secrets_not_inherited(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    # os is banned, so read the env through pandas' own (allowed) machinery is not possible either;
    # we instead check the sandbox env directly by bypassing the static check.
    from analyst import sandbox

    monkeypatch.setattr(sandbox, "check_code", lambda code: [])
    r = run_code("import os\nprint(os.environ.get('GEMINI_API_KEY'))", timeout=30)
    assert r.ok and r.stdout.strip() == "None"


def test_figures_collected(tmp_path):
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.figure(); plt.plot([1, 2, 3])\n"
        "plt.figure(); plt.bar(['a', 'b'], [1, 2])\n"
        "fig = plt.figure(); plt.plot([3, 1]); plt.savefig('own.png'); plt.close(fig)\n"
    )
    r = run_code(code, timeout=60, figures_dir=tmp_path, figure_prefix="step1")
    assert r.ok, r.stderr
    assert [p.name for p in r.figures] == ["step1_1.png", "step1_2.png", "step1_3.png"]
    assert all(p.stat().st_size > 1000 for p in r.figures)


def test_workdir_is_isolated_and_cleaned():
    r = run_code("import pandas as pd\npd.DataFrame({'a':[1]}).to_csv('out.csv')\nprint('ok')", timeout=30)
    assert r.ok
    assert not os.path.exists("out.csv")  # written in the temp dir, not the project


def test_for_llm_summaries():
    rejected = run_code("import os")
    assert "REJECTED" in rejected.for_llm()
    failed = run_code("print('partial')\nraise ValueError('boom')", timeout=30)
    text = failed.for_llm()
    assert "FAILED" in text and "partial" in text and "boom" in text
