import os
import argparse
import shutil

import filetype
from tqdm import tqdm
import whisper
import torchaudio
from moviepy.editor import VideoFileClip


raw_dir = f".{os.sep}raw_data{os.sep}"
raw_files = list(os.walk(raw_dir))[0][2]  # [0][n] n=1为目录, n=2为文件

parser = argparse.ArgumentParser()
parser.add_argument("--speaker", default="speaker")
parser.add_argument("--sample_rate", type=int, default=0)
parser.add_argument("--bit_depth", type=int, default=0)
parser.add_argument("--languages", default="CJE")
parser.add_argument("--whisper_size", default="small", help="what does it mean?")
args = parser.parse_args()

if args.languages == "CJE":
    lang2token = {
        'zh': "[ZH]",
        'ja': "[JA]",
        "en": "[EN]",
    }
elif args.languages == "CJ":
    lang2token = {
        'zh': "[ZH]",
        'ja': "[JA]",
    }
elif args.languages == "C":
    lang2token = {
        'zh': "[ZH]",
        }

# 创建目标目录就建立，又就清空
dest_dir = f".{os.sep}{args.speaker}{os.sep}"
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.mkdir(dest_dir)

# set whisper
model = whisper.load_model("base")
options = dict(beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)

annotation = []
with tqdm(total=100) as pbar:
    for file in raw_files:
        file_name = file.split(".")[0]

        # 如果文件不是音频或视频，跳过
        kind = filetype.guess(raw_dir + file)
        if kind is None:
            continue
        type = kind.mime.split("/")[0]
        if type not in ['audio', 'video']:
            continue

        # 用demucs降噪
        print("\nDenoising with demucs:")
        os.system(f"demucs --two-stems=vocals {raw_dir + file}")
        shutil.copyfile(f".{os.sep}separated{os.sep}htdemucs{os.sep}{file_name}{os.sep}vocals.wav", 
                        f"{raw_dir}denoised_{file_name}.wav")
        shutil.rmtree(f".{os.sep}separated")
        file = f"denoised_{file_name}.wav"

        # 获取 sample_rate 和 bit_depth
        metadata = torchaudio.info(raw_dir + file)
        bit_depth = metadata.bits_per_sample

        result = model.transcribe(raw_dir + file, word_timestamps=True, **transcribe_options)
        segments = result["segments"]

        # 如果识别文字中没有所选语言，跳过此音频文件
        lang = result['language']
        if result['language'] not in list(lang2token.keys()):
            print(f"{lang} not supported, ignoring...\n")
            continue
        
        waveform, sample_rate = torchaudio.load(raw_dir + file, channels_first=True)

        for i, seg in enumerate(result['segments']):
            start_time = seg['start']
            end_time = seg['end']
            text = seg['text']
            text = lang2token[lang] + text.replace("\n", "") + lang2token[lang]
            text = text + "\n"
            wav_seg = wavform[:, int(start_time*sample_rate):int(end_time*sample_rate)]  # waveform是一个二维数组
            wav_seg_name = f"{file_name}_{i}.wav"
            wav_seg_path = dest_dir + wav_seg_name
            annotation.append(wav_seg_path + "|" + text)
            print(f"Transcribed segment: {annotation[-1]}")

            # Merge into one channel
            waveform = waveform.mean(dim=0).unsqueeze(0)

            # Resampling
            if args.sample_rate and args.sample_rate != sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=metadata.sample_rate, new_freq=args.sample_rate)(waveform)
                sample_rate = args.sample_rate

            # Convert to desired bit depth
            if args.bit_depth and args.bit_depth != bit_depth:
                transform = torchaudio.transforms.BitDepth(bits=args.bit_depth)
                waveform = transform(waveform)

            # save wav to dest_dir
            torchaudio.save(wav_seg_path, wav_seg, sample_rate, channels_first=True)

            pbar.update(int(100 / (len(raw_files) * len(result['segments'])) ))

if len(annotation) == 0:
    print("Warning: no audios & videos found!")

with open("./long_character_anno.txt", 'w', encoding='utf-8') as f:
    for line in annotation:
        f.write(line)
