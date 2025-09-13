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
with open('model.json', encoding="utf-8") as f:
  model = json.load(f)
parser = AdaBoostSegmenter(model)
output = parser.predict("一名浙江新昌的茶商說") # [一, 名, 浙江, 新昌, 的, 茶商, 說]
```
