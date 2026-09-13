# Zambezi Voice ASR manifest for La Forge

`manifest.csv` maps every clip of Lozi and Tonga read speech in this
repository to its transcript; `asr-spec.yaml` is the La Forge job that
fine-tunes Whisper on it.

## The dataset

The audio is uploaded to Zakuro storage as the **private** dataset
`zambezi-voice` on [stg.hub.zakuro-ai.com](https://stg.hub.zakuro-ai.com). One
dataset version holds the 10 audio shards, `manifest.csv` (this file's
twin) and a README card. A hub version holds at most 256 files and there are
12,268 clips, so the WAVs are packed into uncompressed tar shards
(`<language>/<code>/audio/<split>-NNN.tar`, at most 512 MiB each).

| language | code | split | clips | hours |
|---|---|---|---:|---:|
| Lozi | `loz` | train | 1,855 | 4.37 |
| Lozi | `loz` | dev | 670 | 0.94 |
| Lozi | `loz` | test | 399 | 0.91 |
| Tonga | `toi` | train | 8,340 | 19.61 |
| Tonga | `toi` | dev | 541 | 1.59 |
| Tonga | `toi` | test | 463 | 1.48 |
| **total** | | | **12,268** | **28.88** |

Not included: Nyanja, whose audio lives in
[unza-speech-lab/zambezi-voice-nyanja](https://github.com/unza-speech-lab/zambezi-voice-nyanja),
and 20 WAVs no transcript references. For 48 clips the TSV's `durationMsec`
disagrees with the WAV header; `duration_ms` is always read from the WAV.

## Columns

| column | meaning |
|---|---|
| `audio` | **Input.** `<shard>#<member>`, relative to the dataset version |
| `transcript` | **Output.** The TSV's `sentence`, surrounding whitespace stripped |
| `split` | `train`, `dev` or `test`, as upstream. La Forge trains on `train`, scores on `dev`, and never reads `test` |
| `language` | ISO 639-3: `loz` or `toi` |
| `duration_ms`, `sample_rate` | From the WAV header (16 kHz mono 16-bit PCM) |
| `offset`, `size`, `sha256` | The clip's bytes inside its shard, and their hash |

One clip is one HTTP Range request: fetch the shard's file route on the hub
with a session or a job grant, **without following the redirect** (storage
refuses a request that also carries a hub credential), then GET the
`Location` with `Range: bytes=<offset>-<offset + size - 1>`.
`python3 scripts/zakuro_hub.py verify` does exactly that.

## Running it on La Forge

`asr-spec.yaml` with `dataset_ref` = the `zambezi-voice` dataset and
`model_ref` = an import of `openai/whisper-large-v3-turbo`. Shona (`sn`) is the
decoder's language token, because Whisper has no Lozi or Tonga. Unsloth trains
a LoRA and merges it, and whisper.cpp quantizes it to ggml `q5_0` for a
`gpu-8gb` device, with the footprint measured and WER/CER reported for f16 vs
quantized.

## Rebuilding

```sh
python3 scripts/zakuro_hub.py build
python3 scripts/zakuro_hub.py push build/hub/zambezi-voice \
    --name zambezi-voice --result build/zambezi-voice.hub.json
python3 scripts/zakuro_hub.py verify --per-shard 5
cp build/hub/zambezi-voice/manifest.csv forge/manifest.csv
```

Shards and manifest are byte-reproducible from the same checkout. `push` and
`verify` need a stg.hub web-session token in `$ZAKURO_HUB_TOKEN` or
`~/.config/zakuro/stg-hub.token`; `zc login` tokens are refused.
