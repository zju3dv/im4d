import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

import azure.cognitiveservices.speech as speechsdk
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3) 
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input text file')
    parser.add_argument('--output', type=str, default=None, help='output audio file')
    parser.add_argument('--voice_type', type=str, default='en-US-AriaNeural', help='output audio file')
    args = parser.parse_args()
    return args

def print_output(speech_synthesis_result, text='Hello, world!'):
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")
                
def parse_lines(input_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    texts = []
    text = ''
    for line in lines:
        if line == '\n':
            texts.append(text)
            text = ''
        else:
            text += line
    if text != '':
        texts.append(text)
    return texts

def gen_audio(text, filename):
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True, filename=filename)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    print_output(speech_synthesis_result)

def main(args):
    speech_config.speech_synthesis_voice_name = args.voice_type
    if args.input[-4:] != '.txt': # Debug only
        gen_audio(args.input, args.output)
        return
    texts = parse_lines(args.input)
    os.makedirs(args.output, exist_ok=True)
    for idx, text in enumerate(texts):
        filename = join(args.output, '{:03d}_{}.mp3'.format(idx, text[:50]))
        gen_audio(text, filename)
if __name__ == '__main__':
    args = parse_args()
    main(args)
### usage ###
# 1. python scripts/azure/tts.py --input "Hello, world!" --output data/outputaudio.mp3
# This will generate an audio file named outputaudio.mp3 in the data folder.
# 2. python scripts/azure/tts.py --input data/neural_scene_chronology.txt --output data/neural_scene_chronology
# This will generate a folder named neural_scene_chronology in the data folder, and each audio file is named as 000_The scene opens on a dark night.mp3, 001_The scene opens on a dark night.mp3, etc.
# One empty line in the input text file means split.
