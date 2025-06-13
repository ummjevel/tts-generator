import gradio as gr
import json
import os
import re
import tempfile
import numpy as np
import soundfile as sf
import resampy

# melo
from melo.api import TTS as MeloTTS

# xttsv2
import torch
# from TTS.api import TTS as xTTSv2
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import XttsAudioConfig
# torch.serialization.add_safe_globals([XttsConfig])
# torch.serialization.safe_globals([XttsAudioConfig])

# 모델/화자 정보 불러오기
with open("config/tts_config.json", "r") as f:
    tts_models = json.load(f)

# 모델 캐시 저장소
model_cache = {}

os.makedirs("output", exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# melo tts
model_melo = MeloTTS(language='KR', device=DEVICE)

# xttsv2
# model_xttsv2 = xTTSv2("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)


def convert_sample_rate(wav_path, target_sr=SAMPLE_RATE, output_path=None):
    # 원본 wav 불러오기
    audio, original_sr = sf.read(wav_path)

    # 필요 시 리샘플링
    if original_sr != target_sr:
        audio_resampled = resampy.resample(audio.T, sr_orig=original_sr, sr_new=target_sr)
        audio_resampled = audio_resampled.T
    else:
        audio_resampled = audio

    # 저장 경로 설정
    if output_path is None:
        base, ext = os.path.splitext(wav_path)
        output_path = f"{base}_{target_sr}hz{ext}"

    # 저장
    sf.write(output_path, audio_resampled, target_sr)
    return output_path


def synthesize_melo(speaker, text):

    speed = 1.2
    speaker_ids = model_melo.hps.data.spk2id
    output_path = "output/melo.wav"
    audio_output = model_melo.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)

    return output_path

# def synthesize_xttsv2(speaker, text, speaker_preview):
#     audio_output = tts.tts(text="Hello world!", speaker_wav=speaker_preview, language="ko")

#     return audio_output


# 음성 합성 함수
def synthesize(model_name, speaker, text, speaker_preview):
    
    audio_output = None

    if model_name == "MeloTTS":
        audio_output = synthesize_melo(speaker, text)
    # elif model_name == "xTTSv2":
    #     audio_output = synthesize_xttsv2(speaker, text, speaker_preview)

    return audio_output

# 전체 합성 파이프라인
def tts_pipeline(input_text, model_name, speaker, speaker_preview):
    # sentences = normalize_text(input_text)
    all_audio = []
    sr = SAMPLE_RATE

    # for sentence in sentences:
    #     audio_output = synthesize(model_name, speaker, sentence)
    #     all_audio.append(audio_output)

    audio_output = synthesize(model_name, speaker, input_text, speaker_preview)

    # full_audio = np.concatenate(all_audio)
    # temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    # sf.write(temp_wav.name, full_audio, sr)

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    result_path = os.path.join('output', temp_wav.name)
    audio_result = convert_sample_rate(audio_output, SAMPLE_RATE, result_path)

    return audio_result

# 화자 목록 및 미리듣기 샘플 갱신
def update_speakers(model_name):
    speakers = tts_models[model_name]["speakers"]
    previews = tts_models[model_name].get("preview_samples", {})
    default_speaker = speakers[0]
    default_preview = previews.get(default_speaker, None)
    return gr.update(choices=speakers, value=default_speaker), default_preview

# 미리듣기 오디오 갱신
def update_preview(model_name, speaker):
    previews = tts_models[model_name].get("preview_samples", {})
    return previews.get(speaker, None)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 음성합성 데모")

    with gr.Row():
        
        # 디폴트 모델 및 화자
        default_model = list(tts_models.keys())[0]
        default_speaker = tts_models[default_model]["speakers"][0]
        default_preview = tts_models[default_model]["preview_samples"][default_speaker]
        # model_dropdown = gr.Dropdown(label="모델 선택", choices=list(tts_models.keys()), value=list(tts_models.keys())[0])
        # speaker_dropdown = gr.Dropdown(label="화자 선택")
        # speaker_preview = gr.Audio(label="화자 미리듣기", type="filepath", interactive=False)
        # Gradio 컴포넌트 초기화
        model_dropdown = gr.Dropdown(
            label="모델 선택",
            choices=list(tts_models.keys()),
            value=default_model
        )

        speaker_dropdown = gr.Dropdown(
            label="화자 선택",
            choices=tts_models[default_model]["speakers"],
            value=default_speaker
        )

        speaker_preview = gr.Audio(
            label="화자 미리듣기",
            type="filepath",
            interactive=False,
            value=default_preview
        )

    text_input = gr.Textbox(lines=4, label="텍스트 입력", placeholder="텍스트를 입력하세요.")
    generate_btn = gr.Button("음성 생성")
    output_audio = gr.Audio(label="합성된 음성", type="filepath")

    # 이벤트 연결
    model_dropdown.change(fn=update_speakers, inputs=model_dropdown, outputs=[speaker_dropdown, speaker_preview])
    speaker_dropdown.change(fn=update_preview, inputs=[model_dropdown, speaker_dropdown], outputs=speaker_preview)
    generate_btn.click(fn=tts_pipeline, inputs=[text_input, model_dropdown, speaker_dropdown, speaker_preview], outputs=output_audio)


# 서버에서 실행
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
