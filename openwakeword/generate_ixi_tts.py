import os
import io
import requests
import json
import uuid
import numpy as np
import resampy
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool

ENDPOINT = 'http://180.228.85.22:5500/inference_tts'
SAVE_DIR = '/home/koo//openWakeWord/my_custom_model/hey_kaygee_M/TTS_BULK'

def save_resampled_wav_from_response(response, save_path, orig_sr=22050, target_sr=16000):
    # 1. response.content는 WAV 바이너리 전체
    byte_stream = io.BytesIO(response.content)

    # 2. WAV 로드 (헤더 자동 파싱, float32로)
    audio, sr = sf.read(byte_stream, dtype='float32')

    # 3. 리샘플링 (22050 → 16000)
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)

    # 4. 저장
    sf.write(save_path, audio, target_sr)

#all_speakers = 'F_MISO, F_INNA, F_BOMI, F_SUA, F_LILI, M_MINWOO, M_WOOJOO, M_BARO,F_MOVIE, M_MOVIE,30_F_TFG, 30_M_TFG,10_F_SHR, 20_M_PJH,40_F_KSH_4544, 40_F_LDW_5312, 40_F_KDG_5495, 40_F_KPJ_5979, 40_F_JJC_7936, 40_F_HDK_5114, 40_F_LCG_5320, 40_F_ESY_5904, 40_F_DHJ_6012, 40_F_CCM_7938, 40_F_SSG_5065, 40_F_PSM_5473, 40_F_SCS_6039, 40_F_KDG_7944, 40_F_LKC_5046, 40_F_KJW_5072, 40_F_KHS_5154, 40_F_HSW_5339, 40_F_YDM_5445, 40_F_LJH_5485, 40_F_LDG_5860, 40_F_HHS_7513, 40_F_CYJ_7955, 40_F_YSJ_5048, 40_F_YSH_5161, 40_F_LSJ_5230, 40_F_KSH_5487, 40_F_KHS_5959, 40_F_KJH_7563, 40_F_KDW_8566, 40_M_KMS_5052, 40_M_LSO_5100, 40_M_SMA_5200, 40_M_BGH_5237, 40_M_KBM_5409, 40_M_KDG_5459, 40_M_KCY_5899, 40_M_KJH_5011, 40_M_JKS_5056, 40_M_JMK_5204, 40_M_LEJ_5242, 40_M_CSM_5428, 40_M_LAK_5471, 40_M_JKH_5843, 40_M_JKH_5035, 40_M_CJH_5133, 40_M_PSJ_5220, 40_M_LSM_5247, 40_M_LSY_5333, 40_M_KMY_5444, 40_M_OEJ_5844, 40_M_KSA_5936, 40_M_KJY_5225, 40_M_LGJ_5260, 40_M_SMH_5951, 40_M_LRY_5085, 40_M_KKH_5295, 40_M_SJH_5343, 40_M_LYJ_5456, 40_M_OJA_5876,0820_M, 0821_M, 0822_M, 0823_M, 0824_M, 0825_M, 0826_M, 0827_M, 0828_M, 0829_M,EA_0180, EA_0185, EA_0186, EA_0188, EA_0189, EA_0191, EA_0192, EA_0193, EA_0199, EA_0200, EA_0204, EA_0206, EA_0208, EA_0209, EA_0210, EA_0212, EA_0219, EA_0220, EA_0221, EA_0223, EA_0228, EA_0231, EA_0232, EA_0234, EA_0235, EA_0236, EA_0237, EA_0240, EA_0243, EA_0245, EA_0246, EA_0250, EA_0251, EA_0252, EA_0253, EA_0258, EA_0259, EA_0260, EA_0263, EA_0264, EA_0266, EA_0267, EA_0270, EA_0272, EA_0273, EA_0274, EA_0275, EA_0276, EA_0282, EA_0286, EA_0287, EA_0288, EA_0291, EA_0292, EA_0293, EA_0295,00_Emotion_A3, 00_Emotion_H3, 00_Emotion_NX, 00_Emotion_S3,00_F_IAN, 00_M_CYH, 00_M_KWON, 00_M_MBC, 00_M_YTN, 00_M_SONG, 00_M_Taylor, 00_N_MOONO, M_CEW'
speakers = 'M_MOVIE, 20_M_PJH,40_F_KSH_4544, 40_F_LDW_5312, 40_F_KDG_5495, 40_F_KPJ_5979, 40_F_JJC_7936, 40_F_HDK_5114, 40_F_LCG_5320, 40_F_ESY_5904, 40_F_DHJ_6012, 40_F_CCM_7938, 40_F_SSG_5065, 40_F_PSM_5473, 40_F_SCS_6039, 40_F_KDG_7944, 40_F_LKC_5046, 40_F_KJW_5072, 40_F_KHS_5154, 40_F_HSW_5339, 40_F_YDM_5445, 40_F_LJH_5485, 40_F_LDG_5860, 40_F_HHS_7513, 40_F_CYJ_7955, 40_F_YSJ_5048, 40_F_YSH_5161, 40_F_LSJ_5230, 40_F_KSH_5487, 40_F_KHS_5959, 40_F_KJH_7563, 40_F_KDW_8566, 40_M_KMS_5052, 40_M_LSO_5100, 40_M_SMA_5200, 40_M_BGH_5237, 40_M_KBM_5409, 40_M_KDG_5459, 40_M_KCY_5899, 40_M_KJH_5011, 40_M_JKS_5056, 40_M_JMK_5204, 40_M_LEJ_5242, 40_M_CSM_5428, 40_M_LAK_5471, 40_M_JKH_5843, 40_M_JKH_5035, 40_M_CJH_5133, 40_M_PSJ_5220, 40_M_LSM_5247, 40_M_LSY_5333, 40_M_KMY_5444, 40_M_OEJ_5844, 40_M_KSA_5936, 40_M_KJY_5225, 40_M_LGJ_5260, 40_M_SMH_5951, 40_M_LRY_5085, 40_M_KKH_5295, 40_M_SJH_5343, 40_M_LYJ_5456, 40_M_OJA_5876,0820_M, 0821_M, 0822_M, 0823_M, 0824_M, 0825_M, 0826_M, 0827_M, 0828_M, 0829_M,EA_0180, EA_0185, EA_0186, EA_0188, EA_0189, EA_0191, EA_0192, EA_0193, EA_0199, EA_0200, EA_0204, EA_0206, EA_0208, EA_0209, EA_0210, EA_0212, EA_0219, EA_0220, EA_0221, EA_0223, EA_0228, EA_0231, EA_0232, EA_0234, EA_0235, EA_0236, EA_0237, EA_0240, EA_0243, EA_0245, EA_0246, EA_0250, EA_0251, EA_0252, EA_0253, EA_0258, EA_0259, EA_0260, EA_0263, EA_0264, EA_0266, EA_0267, EA_0270, EA_0272, EA_0273, EA_0274, EA_0275, EA_0276, EA_0282, EA_0286, EA_0287, EA_0288, EA_0291, EA_0292, EA_0293, EA_0295, 00_M_CYH, 00_M_KWON, 00_M_MBC, 00_M_YTN, 00_M_SONG, 00_M_Taylor, 00_N_MOONO, M_CEW'
speakers = speakers.split(',')
speakers = [sp.strip() for sp in speakers]
with open('TTS_script.txt') as f:
    script = f.readlines()
script = [scr.strip() for scr in script]


# formats = f'<root voice="{speaker}"><speak rate="{rate}" pitch="{pitch}">{script}</speak></root>'

# pitch는 0.9 ~ 1.1 사이로 (변조가 좀 심함)
# rate  는 0.8 ~ 1.3 (speed)
rate = [0.9, 1.0, 1.1, 1.2, 1.3]
pitch = [0.9, 1.0, 1.1]
sampling_rate = 16000


def tts_and_save(speakers):
    for speaker in tqdm(speakers):
        idx = 0
        for text in tqdm(script):
            for speed in rate:
                for pit in pitch:
                    TEXT = f'<root voice="{speaker}"><speak rate="{speed}" pitch="{pit}">{text}</speak></root>'
                    UUID = str(uuid.uuid1().hex)

                    filename = f"{speaker}_{idx:06d}.wav"
                    filename = os.path.join(SAVE_DIR, filename)
                    data = {"text": TEXT, "uuid": UUID, "service":"kws"}
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(ENDPOINT, data=json.dumps(data), headers=headers)

                    response.raise_for_status()
                    save_resampled_wav_from_response(response, filename)
                    idx += 1

if __name__ == '__main__':
    with Pool(processes=16) as pool:
        pool.map(tts_and_save, [(speaker, i) for i, speaker in enumerate(speakers)])