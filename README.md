# NeonAI Tacotron2 TTS Plugin
[Mycroft](https://mycroft-ai.gitbook.io/docs/mycroft-technologies/mycroft-core/plugins) compatible
Mycroft compatible TTS Plugin for Tacotron2 Text-to-Speech.

# Configuration:
```yaml
tts:
    module: neon-tts-plugin-tacotron2
    neon-tts-plugin-tacotron2: {}  # TODO: Any module config
```
# Requirements:
`sudo apt install libsndfile1`

Necessary for recording audio files

## Docker

A docker container using [ovos-tts-server](https://github.com/OpenVoiceOS/ovos-tts-server) is available

You can build and run it locally

```bash
docker build . -t tacotron2
docker run -p 8080:9666 tacotron2
```

use it `http://localhost:8080/synthesize/hello`
