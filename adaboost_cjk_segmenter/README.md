## AdaBoost for Cantonese

Relative to BudouX’s n-gram model, the new [radical](https://en.wikipedia.org/wiki/Chinese_character_radicals)-based AdaBoost model reaches comparable accuracy with under half the model size. The radical of a Chinese character is typically the character's semantic component. Morever, there are only 214 of them in [kRSUnicode](https://en.wikipedia.org/wiki/Kangxi_radicals), making it suitable for lightweight models. The other benefit of using radicals is that, even though the model is trained on only zh-hant data, the radical-based model generalised better, which makes it more suitable to deploy in zh-hant variants such as zh-tw and zh-hk (Cantonese).

**CITYU Test Dataset (zh-hant)**
| Model | F1-Score | Model Size |
|----------|:--------:|:---------:|
| BudouX  | 86.27  | 64 KB  |
| Radical-based  | 85.82  | 31 KB  |
| ICU | 89.46 | 2 MB |

**UDCantonese Dataset (zh-hk)**
| Model | F1-Score | Model Size |
|----------|:--------:|:---------:|
| BudouX  | 73.51  | 64 KB  |
| Radical-based  | 89.76  | 31 KB  |
| [PyCantonese](https://github.com/jacksonllee/pycantonese) | 94.98  | 1.3 MB  |
| ICU | 79.14 | 2 MB |

### Examples

**Test Case 1 (zh-hant)**
| Algorithm | Output |
|----------|:---------|
| Unsegmented | 一名浙江新昌的茶商說正宗龍井產量有限需求量大價格高而貴州茶品質不差混雜在中間根本分不出來 |
| Manually Segmented | 一 . 名 . 浙江 . 新昌 . 的 . 茶商 . 說 . 正宗 . 龍井 . 產量 . 有限 . 需求量 . 大 . 價格 . 高 . 而 . 貴州茶 . 品質 . 不 . 差 . 混雜 . 在 . 中間 . 根本 . 分 . 不 . 出來 |
| Radical-based | 一 . 名 . 浙江 . 新昌 . 的 . 茶商 . 說 . 正宗 . 龍 . 井 . 產量 . 有限 . 需求 . 量 . 大 . 價格 . 高 . 而 . 貴州 . 茶 . 品質 . 不差 . 混雜 . 在 . 中間 . 根本 . 分 . 不 . 出來 |
| BudouX | 一 . 名 . 浙江 . 新昌 . 的 . 茶商 . 說 . 正宗 . 龍井 . 產量 . 有限 . 需求 . 量 . 大 . 價格 . 高 . 而 . 貴州 . 茶品質 . 不差 . 混雜 . 在 . 中間 . 根本 . 分 . 不 . 出來 |
| ICU | 一名 . 浙江 . 新 . 昌 . 的 . 茶商 . 說 . 正宗 . 龍井 . 產量 . 有限 . 需求量 . 大 . 價格 . 高 . 而 . 貴州 . 茶 . 品質 . 不差 . 混雜 . 在中 . 間 . 根本 . 分 . 不出來 |

**Test Case 2 (zh-hk)**
| Algorithm | Output |
|----------|:---------|
| Unsegmented | 點解你唔將呢句說話-點解你同我講，唔同你隔籬嗰啲人講呀？ |
| Manually Segmented | 點解 . 你 . 唔 . 將 . 呢 . 句 . 說話 . - . 點解 . 你 . 同 . 我 . 講 . ， . 唔 . 同 . 你 . 隔籬 . 嗰啲 . 人 . 講 . 呀 . ？ |
| Radical-based | 點解 . 你 . 唔 . 將 . 呢句 . 說話 . - . 點解 . 你 . 同 . 我 . 講 . ， . 唔同 . 你 . 隔籬 . 嗰啲 . 人 . 講 . 呀 . ？ |
| BudouX | 點解你 . 唔 . 將 . 呢句 . 說話 . - . 點解你 . 同 . 我 . 講 . ， . 唔同 . 你 . 隔籬 . 嗰啲人 . 講呀 . ？ |
| ICU | 點 . 解 . 你 . 唔 . 將 . 呢 . 句 . 說話 . - . 點 . 解 . 你 . 同 . 我 . 講 . ， . 唔 . 同 . 你 . 隔 . 籬 . 嗰 . 啲 . 人 . 講 . 呀 . ？ |
| PyCantonese | 點解 . 你 . 唔 . 將 . 呢 . 句 . 說話 . - . 點解 . 你 . 同 . 我 . 講 . ， . 唔同 . 你 . 隔籬 . 嗰啲 . 人 . 講 . 呀 . ？ |

### Usage

Set up the environment using ```pip3 install -r requirements.txt```

```python
import json
from adaboost_cjk_segmenter.helper import AdaBoostSegmenter

with open('model.json', encoding="utf-8") as f:
  model = json.load(f)
segmenter = AdaBoostSegmenter(model)
output = segmenter.predict("一名浙江新昌的茶商說")
```

## Train a model

`train.py` is the complete training entrypoint. It encodes the source data,
trains AdaBoost on the JAX CPU backend, writes checkpoint weights, builds the
final compact model, evaluates every checkpoint, and plots accuracy and F1.

The training file must be non-empty UTF-8 text with one independent sentence
per non-empty line. Mark gold segment boundaries with `▁` (U+2581
LOWER ONE EIGHTH BLOCK), for example:

```text
一名▁浙江▁新昌▁的▁茶商▁說
```

Evaluation uses the CityU, MSR, PKU, and Yue-HK corpora packaged under
`evaluation/datasets`. CityU, MSR, and PKU use whitespace-separated gold words;
Yue-HK uses CoNLL-U. The evaluator retains every token, including punctuation,
and preserves Radaboost's concatenated-corpus boundary labels and
`BinaryMetrics` punctuation adjustment when calculating accuracy and F1.

`evaluate.py` can also evaluate a model independently from training:

```sh
python evaluate.py --dataset cityu --model model.json --language zh
```

### Run locally

Python 3.12 is recommended. From this directory:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python train.py data/train.txt \
  --language zh \
  --iterations 200000 \
  --checkpoint-iterations 50000 100000 150000 200000 \
  --processes 8 \
  --sample-scale 1 \
  --model-scale 1000 \
  --output-dir artifacts
```

The combined CJK profile uses the same invocation with `--language cjk`. A
BudouX fallback language such as Thai also requires gold evaluation data in
the same segmented format:

```sh
python train.py data/thai_train.txt \
  --language thai \
  --evaluation-data data/thai_test.txt \
  --iterations 10000 \
  --checkpoint-iterations 2500 5000 7500 10000 \
  --output-dir artifacts
```

The positional training input may be either a local path or an exact
`gs://bucket/object` URI. A bucket or directory prefix is not accepted as an
input. For fallback languages, `--evaluation-data` has the same local-or-exact-
GCS-object rules. It is required outside `zh`/`cjk` and rejected for those two
Chinese profiles. `--checkpoint-iterations` is required; values must be
positive and no greater than `--iterations`. The final iteration is always
included in evaluation and the graphs, even if it is not an explicit
checkpoint.

The feature profiles are:

- `zh`: reduced Radaboost (`UW2`–`UW5`, `BW2`, `RSRID`, `LSRID`, and `RAD`)
- `cjk`: all 13 BudouX n-gram groups plus the three radical groups
- every other code, including `thai`: the 13 stock BudouX n-gram groups

Encoding and inference share the same bundled ICU4X radical trie. The trie is
loaded lazily and cached once per Python process; entries with value `0` omit
radical features. Neither
`zh` nor `cjk` applies a chunk-length bonus. The reduced groups use Radaboost's
tuned selection rules; the extra BudouX groups in `cjk` use `--feature-thres`.
Fallback languages apply `--feature-thres` to all groups. The default threshold
is `10`; it is unused only for `zh`. The default iteration count is 2,000,000
for `zh`/`cjk` and 10,000 for fallback languages. `--processes` controls
parallel source encoding. Because the trainer normalizes its sample weights,
a uniform `--sample-scale` primarily scales the support counts used for
feature selection; it does not change the relative weight of one sample versus
another. `--model-scale` controls integer truncation in the exported JSON model;
keep the default `1000` to minimize changes from integer rounding.

Successful `zh` and `cjk` runs produce these durable artifacts:

```text
artifacts/
├── weights.txt
├── weights_<iteration>.txt
├── model.json
├── train.log
├── evaluation.log
├── evaluation.json
├── cityu_performance.png
├── msr_performance.png
├── pku_performance.png
├── yue_hk_performance.png
└── combined_performance.png
```

`model.json` always represents the final iteration, rather than the checkpoint
with the best corpus result. Encoded training data and checkpoint model JSON
files are temporary and are not retained or uploaded. The output directory must
be new or empty; a non-empty directory is rejected so files from separate runs
cannot be mixed.
`train.log` records the Radaboost-style dataset, retained-word count, F1, and
accuracy block for all four corpora at every evaluated iteration.
`evaluation.log` provides the same results as a table, and `evaluation.json`
provides structured checkpoint results plus a dedicated final result. Each
per-corpus PNG has accuracy and F1 panels; `combined_performance.png` overlays
all four corpora in both panels.

A fallback-language run replaces those five PNGs with one two-panel graph
named after the normalized language code, such as `thai_performance.png`.
Its logs and JSON contain the single runtime evaluation corpus under that
language name. All other durable artifacts are unchanged.

### Run with Docker

Build the CPU-only image from the repository root:

```sh
docker build -t adaboost-trainer:local adaboost_cjk_segmenter
```

For local files, mount the data read-only and use a separate writable artifact
directory:

```sh
mkdir -p adaboost_cjk_segmenter/artifacts

docker run --rm \
  -v "$PWD/adaboost_cjk_segmenter/data:/data:ro" \
  -v "$PWD/adaboost_cjk_segmenter/artifacts:/artifacts" \
  adaboost-trainer:local \
  /data/train.txt \
  --language zh \
  --iterations 200000 \
  --checkpoint-iterations 50000 100000 150000 200000 \
  --processes 8 \
  --output-dir /artifacts
```

The image fixes JAX to CPU and selects Matplotlib's non-interactive `Agg`
backend. Radical lookup uses the bundled `unihan_radical_trie.json`, so it does
not need a database or runtime internet download; GCS-based runs still need
access to the Cloud Storage API.

To read and write Cloud Storage from a local Docker run, authenticate with
Application Default Credentials (ADC) and mount the gcloud configuration:

```sh
gcloud auth application-default login

docker run --rm \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  adaboost-trainer:local \
  gs://BUCKET_NAME/datasets/DATASET_VERSION/train.txt \
  --language zh \
  --iterations 200000 \
  --checkpoint-iterations 50000 100000 150000 200000 \
  --processes 8 \
  --output-uri gs://BUCKET_NAME/runs/RUN_ID
```

For a Thai or other BudouX fallback run, add an exact evaluation object:

```text
--language thai --evaluation-data gs://BUCKET_NAME/datasets/DATASET_VERSION/test.txt
```

Uploads occur only after training, model loading, evaluation, and plotting all
succeed. Every durable artifact is written directly beneath the specified
output prefix.

### Run on Vertex AI

Push the image to Artifact Registry, then configure a single-replica Vertex AI
Custom Job with no accelerator. The job's service account needs permission to
read the exact training object and create objects under the output prefix.
Vertex supplies ADC automatically; do not put a service-account key in the
image.

This Custom Job configuration uses `baseOutputDirectory`, which makes Vertex
set `AIP_MODEL_DIR` for the container:

```yaml
serviceAccount: TRAINING_SERVICE_ACCOUNT
workerPoolSpecs:
  - machineSpec:
      machineType: c2-standard-16
    replicaCount: "1"
    containerSpec:
      imageUri: REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/adaboost-trainer:GIT_SHA
      args:
        - gs://BUCKET_NAME/datasets/DATASET_VERSION/train.txt
        - --language
        - zh
        - --iterations
        - "200000"
        - --processes
        - "16"
        - --output-dir
        - /tmp/artifacts
        - --checkpoint-iterations
        - "50000"
        - "100000"
        - "150000"
        - "200000"
baseOutputDirectory:
  outputUriPrefix: gs://BUCKET_NAME/runs/RUN_ID
```

Save that configuration as `custom-job.yaml`, substitute every uppercase
placeholder, and submit it with:

```sh
gcloud ai custom-jobs create \
  --display-name=adaboost-cjk-training \
  --region=REGION \
  --config=custom-job.yaml
```

Output destination precedence is explicit: `--output-uri` wins when supplied;
otherwise the trainer uses Vertex's `AIP_MODEL_DIR`; if neither is present,
artifacts remain only in `--output-dir` (default `./artifacts`). With the
`baseOutputDirectory` example above, Vertex sets `AIP_MODEL_DIR` to
`gs://BUCKET_NAME/runs/RUN_ID/model/`, so that is where the artifacts are
uploaded. Choose a unique Cloud Storage prefix for each job so a failed or
repeated job cannot overwrite a prior run.

For a fallback-language Vertex job, add `--evaluation-data` and its exact GCS
object to `containerSpec.args`. The same training service account must be able
to read both the training and evaluation objects.
