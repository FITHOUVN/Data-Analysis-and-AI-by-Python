"""Kiểm tra môi trường: so phiên bản đang có với bộ phiên bản của sách."""
import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

sys.stdout.reconfigure(encoding="utf-8")   # lý do của dòng này: mục 1.8.3

CHUAN = {"Python": "3.11.9", "pandas": "3.0.3", "numpy": "2.4.6",
         "matplotlib": "3.11.0", "seaborn": "0.13.2", "scikit-learn": "1.9.0"}
DANG_CO = {"Python": sys.version.split()[0], "pandas": pd.__version__,
           "numpy": np.__version__, "matplotlib": matplotlib.__version__,
           "seaborn": sns.__version__, "scikit-learn": sklearn.__version__}

for ten, ban_chuan in CHUAN.items():
    ket_luan = "khớp" if DANG_CO[ten] == ban_chuan else "LỆCH"
    print(f"{ten:<13}{DANG_CO[ten]:<10} sách dùng {ban_chuan:<10}{ket_luan}")
