import os
import argparse
import shutil

import filetype

parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", default="/content/sample_data/")
parser.add_argument("--dest_dir", default="/content/drive/MyDrive/")
args = parser.parse_args()

raw_dir = args.dir + os.sep
raw_files = list(os.walk(raw_dir))[0][2]

dest_dir = f".{os.sep}denoised_wav{os.sep}"

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
    os.system(f"demucs --two-stems=vocals {raw_dir + file}")
    shutil.move(f".{os.sep}separated{os.sep}htdemucs{os.sep}{file_name}{os.sep}vocals.wav", 
                    f"{dest_dir+file_name}.wav")
    shutil.rmtree(f".{os.sep}separated")
    os.remove(raw_dir + file)
