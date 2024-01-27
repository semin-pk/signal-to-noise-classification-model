import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
#import matplotlib.pyplot as plt
import numpy as np

# 입력 폴더와 출력 폴더 경로 설정
input_folder = r"C:\Users\tpals\pythonProject2\data\16_bit\spectrogram\snrminus30"
output_folder = r"C:\Users\tpals\pythonProject2\data\16_bit\output_mixed\snrminus30"

# 입력 폴더 내의 음성 파일 목록 가져오기
audio_files = os.listdir(input_folder)

# 출력 폴더가 없다면 생성
os.makedirs(output_folder, exist_ok=True)
a=0
for i, audio_file in enumerate(audio_files):
    # 음성 파일 경로 설정
    audio_path = os.path.join(input_folder, audio_file)

    # 음성 파일 로드
    y, sr = librosa.load(audio_path)

    # 스펙트로그램 생성
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # 스펙트로그램 시각화
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(spectrogram_db, sr=sr)



    # 출력 파일 경로 설정
    output_path = os.path.join(output_folder, audio_file.replace('.wav', '.png'))

    # 스펙트로그램 저장
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    a=a+1
    print(a)
