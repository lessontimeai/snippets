import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd
from scipy.io.wavfile import write

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator([model], cfg)

text = """
Sure, here's a classic English joke for you:

Why -  dont scientists trust atoms?

Because they make up every thing !
"""

sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

ipd.Audio(wav, rate=rate)
audio_data_int = (wav.detach().numpy() * 32767).astype('int16')
write('output.wav', rate, audio_data_int)