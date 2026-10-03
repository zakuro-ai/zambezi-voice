# Training ASR on Zambezi Voice with L'Atelier + Sakura

[L'Atelier](https://github.com/zakuro-ai/sakura) (the engine behind what used to be La Forge's
custom-model flow) turns **one replayable YAML** into a trained, exported and re-scored model.
`task: speech_recognition` trains DeepSpeech2 from scratch with CTC through the
[Sakura](https://github.com/zakuro-ai/sakura) runtime (mixed precision, evaluation overlapped with
training, checkpoint writes in a worker process, resumable). This directory holds the jobs for the
Lozi and Tonga read speech in this repository.

| file | job |
|---|---|
| `loz.yaml` | Lozi: train on `train@loz`, validate on `dev@loz` (4.4 h / 0.9 h) |
| `toi.yaml` | Tonga: train on `train@toi`, validate on `dev@toi` (19.6 h / 1.6 h) |
| `loz-whisper.yaml` | Lozi again, with a pretrained Whisper-small + LoRA (the accuracy route, see below) |
| `run.sh` | run a job on a **local** copy of the dataset |

`test` is never read.

## The dataset

`data.format: asr_manifest` reads `manifest.csv` (`audio`, `transcript`, `split`, `language`, …).
A row's `audio` is `<shard>#<member>`: `offset` / `size` locate the clip's bytes inside an
uncompressed tar shard, and `sha256` pins them. The engine copies nothing: the dataset pin
(`data.sha256`) hashes the manifest and every selected clip, and the trainer re-verifies each clip's
bytes when it reads it, so a replay on different audio fails instead of silently training on it.
Split expressions select rows by split and, after `@`, by language: `train@loz`.

The shards come from `python3 scripts/zakuro_hub.py build` (the same packaging that is published as
the private `zambezi-voice` dataset on the Zakuro hub).

## Run it

```sh
pip install "asr-deepspeech @ git+https://github.com/zakuro-ai/asr" sakura-ml   # CUDA torch first
python3 scripts/zakuro_hub.py build                    # -> build/hub/zambezi-voice
atelier/run.sh loz                                     # -> runs/loz/
```

On the hub the runner presigns the dataset's files and rewrites `data.uri` to a directory in the
engine's copy only; `run.sh` does the same rewrite for a local directory.

`runs/loz/` then contains

* `report.json`: status, `train_seconds`, `billable_seconds`, best `cer`/`wer`, the dataset pin
  and a `runtime_eval` entry: the exported model is **rebuilt from the export alone** and re-scored
  on the validation split;
* `metrics.jsonl`: `loss` (train) and `cer` / `wer` (validation) per epoch, as fractions;
* `resolved.yaml`: the spec with the preset expanded and `data.sha256` filled in. Put that hash in
  the job's `data.sha256` to make a replay verify the data;
* `artifacts/`: `model.safetensors`, `model_config.json`, `labels.json`, `preprocess.json`;
* `checkpoint/`: the best model and rolling resume checkpoints.

Transcribe with the artifact:

```sh
python3 -m sakura.atelier predict runs/loz --file clip.wav     # {"text": "..."}
```

Continue training for more epochs by adding to the spec
`resume_from: {uri: runs/loz/checkpoint}`: `epochs` then means *this many more*.

## Reference runs

Both on the shared RTX 2080 Ti of x399 (one GPU, ~7 GB, `dispatch: process`, `async_eval: true`):

| | Lozi (`loz.yaml`, 30 epochs) | Tonga (`toi.yaml`, 15 epochs) |
|---|---|---|
| train / validation | 1,855 clips (4.4 h) / 670 (0.9 h) | 8,340 clips (19.6 h) / 541 (1.6 h) |
| wall-clock (billable) | 10.1 min | 30.1 min |
| best validation CER / WER | **50.6 % / 95.4 %** | **19.1 % / 79.0 %** |
| export re-scored from `model.safetensors` alone | CER 50.5 % | CER 19.1 % |
| `data.sha256` (pinned in the spec) | `7195d51f…b92358` | `3b2ef9c9…846f9e` |

### Pretrained Whisper vs from scratch (Lozi, same data, same dataset pin)

| | DeepSpeech2 from scratch | Whisper-small + LoRA (`loz-whisper.yaml`, language token `sn`) |
|---|---|---|
| best validation CER / WER | 50.6 % / 95.4 % | **10.1 % / 56.7 %** |
| export re-scored from its files alone | CER 50.5 % | CER 10.1 % |
| wall-clock (shared 2080 Ti) | 10.1 min, 30 epochs | 18.9 min, 3 epochs (batch 8 x accum 2) |
| size of the export | 49 MB | 967 MB |

Pretraining is worth more than anything else on 4.4 h of audio. Whisper has no Lozi, so the decoder is steered with the
Shona token (a related Bantu language, as in `../forge/asr-spec.yaml`); words are still often wrong (WER 57 %) while
characters mostly are right. `model: whisper-large-v3-turbo` is available (needs `gradient_checkpointing: true` on a
12 GB card; not run here).

Read the DeepSpeech2 numbers above as a **working pipeline and baselines, not good models**. Lozi: a from-scratch
character-level CTC model on 4.4 h of audio memorises its training set (train loss 0.8 against a
validation CER of 51 %). Tonga has 4.5x the data and reaches CER 19 %, but WER stays at 79 %. More data, SpecAugment
(`spec_augment: true`), or a pretrained model (the Whisper path in `../forge/`) are the routes to
accuracy; the point of this job is that the same replayable spec trains, exports, re-scores and
resumes anywhere the Atelier engine runs.

## Pitfalls this surfaced (all fixed upstream)

* A CSV label file cannot hold a space, so the alphabet silently lost it and every transcript with
  a space indexed past the output layer: the loss froze with no error. `asr-deepspeech` now takes the
  alphabet as a list, and the backend checks the class count.
* The engine's `torch.use_deterministic_algorithms(True)` makes PyTorch run CTC loss on cuDNN, whose
  backward returns NaN for a whole batch when one utterance is unalignable. The DeepSpeech backend
  opts out of that flag, and the trainer drops unalignable utterances before the loss.
* A run with no finite gradient step, or without one completed evaluation, now fails instead of
  reporting `done`.

## Tuning

`overrides:` is merged over the preset (`speech_recognition/ctc-fast@1`: 3 x 512 bidirectional GRU,
AdamW, mixed precision, `dispatch: process`, `async_eval: true`, resume checkpoint at most every 60 s).
`backend_config:` is merged last for any trainer key. Set `stop_cer: 30` (CER in percent) in `overrides`
to stop at a target CER instead of a fixed epoch count.

## Whisper (La Forge)

`../forge/asr-spec.yaml` is the older La Forge job that LoRA-fine-tunes Whisper on the same manifest
and ships ggml. Both paths read the same `forge/manifest.csv` columns; the Atelier path trains a
small from-scratch model (sized for a shared 8 GB GPU), the Whisper path adapts a large pretrained
one. The Atelier `whisper` backend (above) is the same idea inside the engine: replayable spec, hash-pinned data, merged `safetensors` export re-scored from its own files.
