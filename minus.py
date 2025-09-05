import argparse
import torch
import torchaudio
from typing import Tuple
import numpy as np

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

def subtract_audio_files(
    audio1_path: str,
    audio2_path: str,
    output_path: str,
    target_sample_rate: int = 44100,
    target_channels: int = 1,
    normalize_output: bool = True,
    verbose: bool = True
):
    """
    读取两个音频文件，相减后保存到输出文件
    
    Args:
        audio1_path: 第一个音频文件路径
        audio2_path: 第二个音频文件路径
        output_path: 输出音频文件路径
        target_sample_rate: 目标采样率
        target_channels: 目标通道数
        normalize_output: 是否对输出进行归一化
        verbose: 是否显示详细信息
    """
    
    # 加载并预处理音频
    audio1 = load_and_preprocess_audio(
        audio1_path, target_sample_rate, target_channels, verbose
    )
    audio2 = load_and_preprocess_audio(
        audio2_path, target_sample_rate, target_channels, verbose
    )
    
    # 对齐音频长度
    audio1, audio2 = align_audio_length(audio1, audio2)
    
    if verbose:
        print(f"Audio 1 shape after alignment: {audio1.shape}")
        print(f"Audio 2 shape after alignment: {audio2.shape}")
    
    # 音频相减
    result_audio = audio1 - audio2
    
    if verbose:
        print(f"Result audio shape: {result_audio.shape}")
        print(f"Result audio range: [{result_audio.min():.4f}, {result_audio.max():.4f}]")
    
    # 可选：归一化输出
    if normalize_output:
        max_val = torch.max(torch.abs(result_audio))
        if max_val > 0:
            result_audio = result_audio / max_val
            if verbose:
                print(f"Normalized result audio range: [{result_audio.min():.4f}, {result_audio.max():.4f}]")
    
    # 保存结果
    torchaudio.save(output_path, result_audio, target_sample_rate)
    
    if verbose:
        print(f"Result saved to: {output_path}")
        print(f"Sample rate: {target_sample_rate} Hz")
        print(f"Channels: {result_audio.shape[0]}")
        print(f"Duration: {result_audio.shape[1] / target_sample_rate:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtract two audio files and save the result")
    parser.add_argument("audio1", help="First audio file (e.g., a.wav)")
    parser.add_argument("audio2", help="Second audio file (e.g., b.wav)")
    parser.add_argument("output", help="Output audio file (e.g., result.wav)")
    parser.add_argument("--sample_rate", type=int, default=44100, 
                       help="Target sample rate (default: 44100)")
    parser.add_argument("--channels", type=int, default=1, 
                       help="Target number of channels (default: 1)")
    parser.add_argument("--no_normalize", action="store_true",
                       help="Disable output normalization")
    
    args = parser.parse_args()
    
    subtract_audio_files(
        audio1_path=args.audio1,
        audio2_path=args.audio2,
        output_path=args.output,
        target_sample_rate=args.sample_rate,
        target_channels=args.channels,
        normalize_output=not args.no_normalize,
        verbose=True
    )