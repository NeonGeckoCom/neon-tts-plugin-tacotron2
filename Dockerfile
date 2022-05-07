FROM python:3.8-buster

RUN apt-get update -y && apt-get install -y libsndfile1 python3-pip

COPY . /tmp/neon-tts-plugin-tacotron2

RUN pip3 install ovos-tts-server==0.0.2
RUN pip3 install /tmp/neon-tts-plugin-tacotron2

ENTRYPOINT ovos-tts-server --engine neon-tts-plugin-tacotron2