from pathlib import Path

from wm import App

from birdclef_2026.experiments.baseline import BirdCLEFBaseline
from birdclef_2026.experiments.finetune import BirdCLEFFinetune

app = App.from_pyproject(Path(__file__).parent.parent.parent)
app.register(BirdCLEFBaseline)
app.register(BirdCLEFFinetune)
