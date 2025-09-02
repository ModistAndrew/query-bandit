> ### Please consider giving back to the community if you have benefited from this work.
>
> If you've **benefited commercially from this work**, which we've poured significant effort into and released under permissive licenses, we hope you've found it valuable! While these licenses give you lots of freedom, we believe in nurturing a vibrant ecosystem where innovation can continue to flourish.
>
> So, as a gesture of appreciation and responsibility, we strongly urge commercial entities that have gained from this software to consider making voluntary contributions to music-related non-profit organizations of your choice. Your contribution directly helps support the foundational work that empowers your commercial success and ensures open-source innovation keeps moving forward.
>
> Some suggestions for the beneficiaries are provided [here](https://github.com/the-secret-source/nonprofits). Please do not hesitate to contribute to the list by opening pull requests there.

---


<div align="center">
	<img src="assets/banquet-logo.png">
</div>

# Banquet: A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems

Repository for **A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems** 
by Karn N. Watcharasupat and Alexander Lerch. [arXiv](https://arxiv.org/abs/2406.18747)

> Despite significant recent progress across multiple subtasks of audio source separation, few music source separation systems support separation beyond the four-stem vocals, drums, bass, and other (VDBO) setup. Of the very few current systems that support source separation beyond this setup, most continue to rely on an inflexible decoder setup that can only support a fixed pre-defined set of stems. Increasing stem support in these inflexible systems correspondingly requires increasing computational complexity, rendering extensions of these systems computationally infeasible for long-tail instruments. In this work, we propose Banquet, a system that allows source separation of multiple stems using just one decoder. A bandsplit source separation model is extended to work in a query-based setup in tandem with a music instrument recognition PaSST model. On the MoisesDB dataset, Banquet, at only 24.9 M trainable parameters, approached the performance level of the significantly more complex 6-stem Hybrid Transformer Demucs on VDBO stems and outperformed it on guitar and piano. The query-based setup allows for the separation of narrow instrument classes such as clean acoustic guitars, and can be successfully applied to the extraction of less common stems such as reeds and organs.

For the Cinematic Audio Source Separation model, Bandit, see [this repository](https://github.com/kwatcharasupat/bandit).

## Inference

```bash
git clone https://github.com/kwatcharasupat/query-bandit.git
cd query-bandit
export CONFIG_ROOT=./config
python -m debugpy --listen 5678 --wait-for-client \
train.py inference_byoq \
  results/ev-pre-aug.ckpt \
  input/491c1ff5-1e7b-4046-8029-a82d4a8aefb4.wav \
  input/491c1ff5-1e7b-4046-8029-a82d4a8aefb4_bass.wav \
  output/491c1ff5-1e7b-4046-8029-a82d4a8aefb4_bass.wav \
  --batch_size=12 \
  --use_cuda=true

python \
train.py inference_byoq \
  logs/vdb-pre-aug/e2e/HBRPOI/lightning_logs/version_0/checkpoints/last.ckpt \
  input/491c1ff5-1e7b-4046-8029-a82d4a8aefb4.wav \
  input/491c1ff5-1e7b-4046-8029-a82d4a8aefb4_piano.wav \
  output/491c1ff5-1e7b-4046-8029-a82d4a8aefb4_piano_vdb-pre-aug.wav \
  --batch_size=12 \
  --use_cuda=true

python train.py inference_test_folder   results/ev-pre-aug.ckpt   /inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/data/karaoke_converted/test output/karaoke bass  --batch_size=30   --use_cuda=true --input_name=mixture

export CONFIG_ROOT=./config
export DATA_ROOT=/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/data
export DATA_ROOT=/dev/shm
export DATA_ROOT=/inspire/ssd/project/multilingualspeechrecognition/public
export LOG_ROOT=./logs
export CUDA_VISIBLE_DEVICES=0
python \
train.py train \
  expt/setup-c/bandit-everything-query-pre-d-aug-bal.yml \
  --ckpt_path=logs/ev-pre-aug-bal/e2e/HBRPOI/lightning_logs/version_1/checkpoints/last.ckpt
```
Batch size of 12 _usually_ fits on a RTX 4090.

### Model weights
Model weights are available on Zenodo [here](https://zenodo.org/records/13694558).
If you are not sure, use `ev-pre-aug.ckpt`.

## Citation
```
@inproceedings{Watcharasupat2024Banquet,
  title = {A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems},
  booktitle = {To Appear in the Proceedings of the 25th International Society for Music Information Retrieval},
  author = {Watcharasupat, Karn N. and Lerch, Alexander},
  year = {2024},
  month = {nov},
  eprint = {2406.18747},
  address = {San Francisco, CA, USA},
}
```
