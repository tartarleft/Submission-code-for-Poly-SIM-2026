# Code Submission: Router-Only v2

Standalone inference package. All model weights and configs included.

## Quick Start

```bash
cd Submission-code-for-Poly-SIM-2026
huggingface-cli download tartarleft/polysim-2026-model --local-dir ./
# Single GPU
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --test_home /path/to/test_data \
    --english_csv /path/to/v1_test_English.csv \
    --urdu_csv /path/to/v1_test_Urdu.csv \
    --output_dir submission
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--test_home` | ✅ | Root directory of test data. CSV `voices`/`faces` columns contain relative paths that are joined with this directory. |
| `--english_csv` | ✅ | Path to English test CSV file. |
| `--urdu_csv` | ✅ | Path to Urdu test CSV file. |
| `--checkpoint` | No (default: `./epoch_030.pt`) | Path to model checkpoint. |
| `--output_dir` | No (default: `submission`) | Output directory for submission CSVs. |
| `--gpu` | No (default: `0`) | GPU ID. |

## test_home Directory Layout

`test_home` is the root directory of test data. The `voices` and `faces` columns in the CSV files contain **paths relative to `test_home`**. During inference, full paths are resolved as:

```
${test_home}/${row['voices']}   # full path to audio file
${test_home}/${row['faces']}    # full path to face image
```

For example, if the CSV contains:

```csv
key,voices,faces
t5M7dziYVY,test/v1/voices/English/00001.wav,test/v1/faces/English/00001.jpg
```

Then `test_home` should contain:

```
test_home/
├── test/v1/voices/English/00001.wav
└── test/v1/faces/English/00001.jpg
```

## Multi-GPU

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --test_home /path/to/test_data \
    --english_csv /path/to/v1_test_English.csv \
    --urdu_csv /path/to/v1_test_Urdu.csv \
    --output_dir /tmp/sub_g0 --gpu 0 &

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --test_home /path/to/test_data \
    --english_csv /path/to/v1_test_English.csv \
    --urdu_csv /path/to/v1_test_Urdu.csv \
    --output_dir /tmp/sub_g1 --gpu 1 &

wait

# Merge (each process writes all samples for its shard)
mkdir -p submission
python -c "
import pandas as pd
for name in ['submission_v1_test_English_English', 'submission_v1_test_English_Urdu']:
    dfs = [pd.read_csv(f'/tmp/sub_g{i}/{name}.csv') for i in range(2)]
    merged = dfs[0].set_index('key').combine_first(dfs[1].set_index('key')).reset_index()
    merged.to_csv(f'submission/{name}.csv', index=False)
    print(f'submission/{name}.csv: {len(merged)} rows')
"
```

## Output

Two submission files are generated:

| File | Columns |
|------|---------|
| `submission_v1_test_English_English.csv` | `key`, `p3` (fusion), `p4` (audio only) |
| `submission_v1_test_English_Urdu.csv` | `key`, `p5` (fusion), `p6` (audio only) |

## Expected Results

| Language | Fusion (p3/p5) | Audio Only (p4/p6) |
|----------|----------------|---------------------|
| English  | p3: **99.93%** (1520/1521) | p4: 97.50% (1483/1521) |
| Urdu     | p5: **100.00%** (1623/1623) | p6: 98.83% (1604/1623) |

## Dependencies

Install the wespeaker conda environment first:

```bash
conda create -n wespeaker python=3.9
conda activate wespeaker
pip install -r requirements.txt
```

`requirements.txt` is exported from the wespeaker conda environment:


## Directory Structure

```
code_submission/
├── epoch_030.pt              # Model checkpoint (all weights included)
├── inference.py              # Main inference script
├── config.py                 # Experiment config
├── models/
│   ├── __init__.py
│   ├── router_only.py        # Router-Only model
│   ├── audio_encoder.py      # Audio encoder wrapper
│   └── face_encoder.py       # Face encoder wrapper
├── face/models/              # Face model (IResNet-18)
│   ├── __init__.py
│   ├── base/
│   └── iresnet/
│       ├── model.py
│       └── configs/v1_ir18.yaml
├── wespeaker/                # Audio model dependencies
│   ├── frontend/
│   ├── models/
│   └── utils/
├── hf_cache/                 # w2v-bert HuggingFace cache (offline)
│   └── hub/models--facebook--w2v-bert-2.0/
└── README.md
```

## Notes

- `epoch_030.pt` contains ALL model weights (audio encoder, face encoder, classifiers, adapters, router). No external pretrained checkpoints needed.
- w2v-bert model files are included in `hf_cache/` for offline inference.
- The model uses ~600MB GPU memory at inference time.
