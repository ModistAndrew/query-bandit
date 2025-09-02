import argparse
import torch
import torchaudio
from typing import Tuple
import museval
from museval.aggregate import TrackStore

from core.metrics.snr import safe_signal_noise_ratio, safe_scale_invariant_signal_noise_ratio
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio, SignalDistortionRatio, ScaleInvariantSignalDistortionRatio

def load_and_preprocess_audio(
    audio_path: str, 
    target_sample_rate: int = 44100,
    target_channels: int = 1,
    verbose: bool = True
) -> torch.Tensor:
    # 加载音频
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    
    if verbose:
        print(f"Loaded audio: {audio_path}")
        print(f"  Original shape: {waveform.shape}")
        print(f"  Original sample rate: {orig_sample_rate} Hz")
        print(f"  Duration: {waveform.shape[1] / orig_sample_rate:.2f} seconds")
        print(f"  Target sample rate: {target_sample_rate} Hz")
        print(f"  Target channels: {target_channels}")
    
    # 重采样处理
    if orig_sample_rate != target_sample_rate:
        if verbose:
            print(f"  Resampling: {orig_sample_rate} Hz -> {target_sample_rate} Hz")
        
        resampler = torchaudio.transforms.Resample(
            orig_sample_rate, target_sample_rate
        )
        waveform = resampler(waveform)
        
        if verbose:
            print(f"  After resampling shape: {waveform.shape}")
            print(f"  New duration: {waveform.shape[1] / target_sample_rate:.2f} seconds")
    
    # 通道处理
    current_channels = waveform.shape[0]
    
    if current_channels > target_channels:
        if verbose:
            print(f"  Downmixing: {current_channels} channels -> {target_channels} channel(s)")
            print(f"  Using mean averaging for downmixing")
        
        assert target_channels == 1, "Downmixing only supported to mono"
        waveform = waveform.mean(dim=0, keepdim=True)
        
        if verbose:
            print(f"  After downmixing shape: {waveform.shape}")
    
    elif current_channels < target_channels:
        if verbose:
            print(f"  Upmixing: {current_channels} channel(s) -> {target_channels} channels")
            print(f"  Repeating single channel data")
        
        assert waveform.shape[0] == 1, "Upmixing only supported from mono"
        waveform = waveform.repeat(target_channels, 1)
        
        if verbose:
            print(f"  After upmixing shape: {waveform.shape}")
    
    else:
        if verbose:
            print(f"  No channel conversion needed (already {target_channels} channels)")
    
    # 最终信息
    if verbose:
        print(f"  Final shape: {waveform.shape}")
        print(f"  Final sample rate: {target_sample_rate} Hz")
        print(f"  Final duration: {waveform.shape[1] / target_sample_rate:.2f} seconds")
        print("-" * 50)
    
    return waveform

def align_audio_length(
    audio1: torch.Tensor, 
    audio2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_length = min(audio1.shape[1], audio2.shape[1])
    audio1_aligned = audio1[:, :min_length]
    audio2_aligned = audio2[:, :min_length]
    return audio1_aligned, audio2_aligned

def calculate_audio_snr(
    pred_audio_path: str,
    target_audio_path: str,
):
    target_audio, target_sr = torchaudio.load(target_audio_path)
    target_sample_rate = target_sr
    target_channels = target_audio.shape[0]

    pred_audio = load_and_preprocess_audio(
        pred_audio_path, target_sample_rate, target_channels
    )
    target_audio = load_and_preprocess_audio(
        target_audio_path, target_sample_rate, target_channels
    )
    
    pred_audio, target_audio = align_audio_length(pred_audio, target_audio)
    
    SDR, ISR, SIR, SAR = museval.evaluate(pred_audio.T.unsqueeze(0), target_audio.T.unsqueeze(0))
    data = TrackStore(track_name="test")
    data.add_target(target_name="target", values={"SDR": SDR[0].tolist(), "SIR": SIR[0].tolist(), "SAR": SAR[0].tolist(), "ISR": ISR[0].tolist()})
    print("MusEval results: ")
    print(data)
    
    snr_value = safe_signal_noise_ratio(pred_audio, target_audio)
    print("SNR", snr_value)
    
    sisnr_value = safe_scale_invariant_signal_noise_ratio(pred_audio, target_audio)
    print("SI-SNR", sisnr_value)
    
    torch_snr = SignalNoiseRatio()
    torch_snr_value = torch_snr(pred_audio, target_audio)
    print("Torch SNR", torch_snr_value)
    
    torch_sisnr = ScaleInvariantSignalNoiseRatio()
    torch_sisnr_value = torch_sisnr(pred_audio, target_audio)
    print("Torch SI-SNR", torch_sisnr_value)
    
    torch_sdr = SignalDistortionRatio()
    torch_sdr_value = torch_sdr(pred_audio, target_audio)
    print("Torch SDR", torch_sdr_value)
    
    torch_sisdr = ScaleInvariantSignalDistortionRatio()
    torch_sisdr_value = torch_sisdr(pred_audio, target_audio)
    print("Torch SI-SDR", torch_sisdr_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metric")
    parser.add_argument("a", help="a.wav")
    parser.add_argument("b", help="b.wav")
    
    args = parser.parse_args()
    
    calculate_audio_snr(args.a, args.b)