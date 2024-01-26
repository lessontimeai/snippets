# Load FastPitch
from nemo.collections.tts.models import FastPitchModel
spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")

# Load vocoder
from nemo.collections.tts.models import HifiGanModel
model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
