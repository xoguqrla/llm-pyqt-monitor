import os

# 기본을 v3로
_ver = (os.getenv("VIZ_VERSION") or "v3").lower()

if _ver in ("3", "v3"):
    from .analysis_visualizer_v3 import AnalysisVisualizer
elif _ver in ("2", "v2"):
    from .analysis_visualizer_v2 import AnalysisVisualizer
else:
    from .analysis_visualizer_v1 import AnalysisVisualizer

__all__ = ["AnalysisVisualizer"]
