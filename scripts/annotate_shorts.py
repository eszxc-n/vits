import os
import argparse

import torchaudio
import whisper
import torch
from tqdm import tqdm


def transcribe_gpu(audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(f"GPU: Language - {lang}; Text - {result.text}")
    return lang, result.text

def transcribe_cpu(audio_path):
    result = model.transcribe(audio_path)
    lang = result["language"]
    text = result["text"]
    print(f"Language: {lang}; Text: {text}")
    return lang, text


# script开始
parser = argparse.ArgumentParser()
parser.add_argument("audio_dir")
parser.add_argument("--languages", default="")
parser.add_argument("--speaker", default="speaker")
parser.add_argument("--sample_rate", type=int, default=0)
parser.add_argument("--bit_depth", type=int, default=0)
parser.add_argument("--whisper_size", default="medium")
args = parser.parse_args()

model = whisper.load_model(args.whisper_size)
is_gpu = torch.cuda.is_available()

audio_dir = args.audio_dir
filelist = list(os.walk(audio_dir))[0][2]

speaker = args.speaker
if not os.path.exists(speaker):
    os.mkdir(speaker)

lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
}


annotations = []
for i, wavfile in enumerate(filelist):
    if not wavfile.endswith(".wav"):
        continue
    
    try:
        # get bit_depth
        metadata = torchaudio.info(audio_dir + wavfile)
        # load file
        waveform, sample_rate = torchaudio.load(audio_dir + wavfile)
        # merge channels into 1
        waveform = waveform.mean(dim=0).unsqueeze(0)
        # Resampling
        if args.sample_rate and args.sample_rate != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.sample_rate)(waveform)
            sample_rate = args.sample_rate
        # Convert bit_depth
        if args.bit_depth and args.bit_depth != metadata.bits_per_sample:
            transform = torchaudio.transforms.BitDepth(bits=args.bit_depth)
            waveform = transform(waveform)
        # save wav file
        save_path = speaker + os.sep + f"processed_{i}.wav"
        torchaudio.save(save_path, waveform, sample_rate)
        
        # transcribe text
        if is_gpu:
            lang, text = transcribe_gpu(save_path)
        else:
            lang, text = transcribe_cpu(save_path)
        if lang not in list(lang2token.keys()):
            print(f"{lang} not supported, ignoring\n")
            continue
        # 是否加语言标签
        if args.languages:
            lang_label = lang2token[lang]
        else:
            lang_label = ""
        text = lang_label + text + lang_label + "\n"
        annotations.append(save_path + "|" + text)
    
    except:
        print("error")

if len(annotations) == 0:
    print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
    print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
with open(f"{speaker}_text_train.txt", 'w', encoding='utf-8') as f:
    for line in annotations:
        f.write(line)
