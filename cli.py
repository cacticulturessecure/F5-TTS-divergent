import argparse
import os
import torch
import torchaudio
from einops import rearrange
from vocos import Vocos

from model import CFM, DiT, UNetT
from model.utils import load_checkpoint, get_tokenizer, convert_char_to_pinyin

# Constants
TARGET_SAMPLE_RATE = 24000
N_MEL_CHANNELS = 100
HOP_LENGTH = 256
TARGET_RMS = 0.1
DATASET_NAME = "Emilia_ZH_EN"
TOKENIZER = "pinyin"
REF_AUDIO_DIR = "tests"
REF_AUDIO_FILE = "test.wav"

def load_model(model_name, device):
    if model_name == "F5-TTS":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        ckpt_path = "ckpts/F5TTS_Base/model_1200000.pt"
    elif model_name == "E2-TTS":
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        ckpt_path = "ckpts/E2TTS_Base/model_1200000.pt"
    else:
        raise ValueError("Invalid model name")

    vocab_char_map, vocab_size = get_tokenizer(DATASET_NAME, TOKENIZER)
    
    model = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=N_MEL_CHANNELS
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=TARGET_SAMPLE_RATE,
            n_mel_channels=N_MEL_CHANNELS,
            hop_length=HOP_LENGTH,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema=True)
    return model

def load_vocos():
    return Vocos.from_pretrained("charactr/vocos-mel-24khz")

def process_audio(audio_path, device):
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio.to(device)

def synthesize_wav_file(model, vocos, device):
    audio_path = os.path.join(REF_AUDIO_DIR, REF_AUDIO_FILE)
    audio = process_audio(audio_path, device)
    
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=audio,
            text=["This is a test sentence for synthesis."],
            duration=audio.shape[-1] // HOP_LENGTH,
            steps=32,
            cfg_strength=2.0,
        )
    
    generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    
    output_path = os.path.join(REF_AUDIO_DIR, f"synthesized_{REF_AUDIO_FILE}")
    torchaudio.save(output_path, generated_wave, TARGET_SAMPLE_RATE)

def get_user_input():
    while True:
        text = input("Enter your text (8-16 words): ")
        words = text.split()
        if 8 <= len(words) <= 16:
            return text
        print("Please enter between 8 and 16 words.")

def generate_output_audio(model, vocos, text, device):
    ref_audio_path = os.path.join(REF_AUDIO_DIR, REF_AUDIO_FILE)
    audio = process_audio(ref_audio_path, device)
    
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=audio,
            text=[text],
            duration=audio.shape[-1] // HOP_LENGTH,
            steps=32,
            cfg_strength=2.0,
        )
    
    generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    
    output_path = os.path.join(REF_AUDIO_DIR, "completed_output.wav")
    torchaudio.save(output_path, generated_wave, TARGET_SAMPLE_RATE)

def main():
    parser = argparse.ArgumentParser(description="F5-TTS CLI")
    parser.add_argument("--model", choices=["F5-TTS", "E2-TTS"], default="F5-TTS", help="Choose TTS model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(args.model, device)
    print("Loading vocoder...")
    vocos = load_vocos()

    print(f"Synthesizing {REF_AUDIO_FILE}...")
    synthesize_wav_file(model, vocos, device)
    print(f"Wav file synthesized successfully. Saved as 'synthesized_{REF_AUDIO_FILE}' in the {REF_AUDIO_DIR} directory.")

    user_text = get_user_input()

    print("Generating output audio...")
    generate_output_audio(model, vocos, user_text, device)
    print(f"Output audio generated successfully. Saved as 'completed_output.wav' in the {REF_AUDIO_DIR} directory.")

if __name__ == "__main__":
    main()
