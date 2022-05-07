# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2022 Neongecko.com Inc.
# Contributors: Daniel McKnight, Guy Daniels, Elon Gasper, Richard Leeds,
# Regina Bloomstine, Casimiro Ferreira, Andrii Pernatii, Kirill Hrymailo
# BSD-3 License
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional

import soundfile as sf
from ovos_plugin_manager.templates.tts import TTS, TTSValidator
from ovos_utils.log import LOG
from ovos_utils.metrics import Stopwatch
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel


class Tacotron2TTS(TTS):
    langs = {
        "en-us": {
            "mel": "tensorspeech/tts-tacotron2-ljspeech-en",
            "vocoder": "tensorspeech/tts-mb_melgan-ljspeech-en"
        },
        "pl-pl": {
            "mel": "NeonBohdan/tts-tacotron2-ljspeech-pl",
            "vocoder": "tensorspeech/tts-mb_melgan-ljspeech-en"
        }
    }

    def __init__(self, lang="en-us", config=None):
        super(Tacotron2TTS, self).__init__(lang, config, Tacotron2TTSValidator(self),
                                           audio_ext="wav",
                                           ssml_tags=["speak"])
        self._init_model()

    def get_tts(self, sentence: str, output_file: str, speaker: Optional[dict] = None):
        stopwatch = Stopwatch()
        speaker = speaker or dict()

        # TODO: speaker params are optionally defined and should be handled whenever defined
        # # Read utterance data from passed configuration
        # request_lang = speaker.get("language",  self.lang)
        # request_gender = speaker.get("gender", "female")
        # request_voice = speaker.get("voice")

        # TODO: Below is an example of a common ambiguous language code; test and implement or remove
        # # Catch Chinese alt code
        # if request_lang.lower() == "zh-zh":
        #     request_lang = "cmn-cn"

        with stopwatch:
            self._run_model(sentence=sentence, output_file=output_file)

        LOG.debug(f"TTS Synthesis time={stopwatch.time}")

        return output_file, None

    def _init_model(self):
        langParams = self.langs[self.lang]
        mevName = langParams["mel"]
        vocoderName = langParams["vocoder"]

        # initialize tacotron2 model.
        self.model = TFAutoModel.from_pretrained(mevName)

        # initialize vocoder
        self.vocoder = TFAutoModel.from_pretrained(vocoderName)

        # processor
        self.text_processor = AutoProcessor.from_pretrained(mevName)

    def _run_model(self, sentence: str, output_file: str):
        input_ids = self.text_processor.text_to_sequence(sentence)

        decoder_output, mel_outputs, stop_token_prediction, alignment_history = self.model.inference(
            input_ids=[input_ids],
            input_lengths=[len(input_ids)],
            speaker_ids=[0],
        )

        # vocoder inference
        audio = self.vocoder.inference(mel_outputs)[0, :, 0]

        # save to file
        sf.write(output_file, audio, 22050, "PCM_16")


class Tacotron2TTSValidator(TTSValidator):
    def __init__(self, tts):
        super(Tacotron2TTSValidator, self).__init__(tts)

    def validate_lang(self):
        pass

    def validate_dependencies(self):
        # TODO: Optionally check dependencies or raise
        pass

    def validate_connection(self):
        # TODO: Optionally check connection to remote service or raise
        pass

    def get_tts_class(self):
        return Tacotron2TTS
