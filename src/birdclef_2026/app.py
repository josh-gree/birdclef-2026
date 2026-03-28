from pathlib import Path

from wm import App

from birdclef_2026.experiments.attn_sed import BirdCLEFAttnSED
from birdclef_2026.experiments.baseline import BirdCLEFBaseline
from birdclef_2026.experiments.perch_mlp import BirdCLEFPerchMLP

app = App.from_pyproject(Path(__file__).parent.parent.parent)
app.register(BirdCLEFBaseline)
app.register(BirdCLEFAttnSED)
app.register(BirdCLEFPerchMLP)
