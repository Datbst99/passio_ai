import os

import torch
from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts

CHECKPOINT_DIR = 'model/'
USE_DEEPSPEED = True

def initialization():
    print("Loading model...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xtts_config = os.path.join(CHECKPOINT_DIR, "config.json")
    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, use_deepspeed=USE_DEEPSPEED, eval=True)

    if torch.cuda.is_available():
        model.cuda()

    return model