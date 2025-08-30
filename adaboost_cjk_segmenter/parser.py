from cihai.core import Cihai
c = Cihai()
if not c.unihan.is_bootstrapped:
  c.unihan.bootstrap()

def get_radical(ch1: str):
    char1 = c.unihan.lookup_char(ch1).first()
    if char1 is None:
        return 0
    else:
        r1 = char1.kRSUnicode.split(" ")[0]
        if '\'' in r1:
            return r1.split('\'')[0]
        else:
            return r1.split('.')[0]

class AdaBoostSegmenter:
    def __init__(self, model):
        self.model = model
    
    def predict(self, sentence):
        if sentence == '':
            return []
        chunks = [sentence[0]]
        base_score = -sum(sum(g.values()) for g in self.model.values()) * 0.5

        for i in range(1, len(sentence)):
            score = base_score
            L = len(chunks[-1])
            score += 32**L
            rad4 = get_radical(sentence[i])
            if rad4:
                score += self.model.get('RSRID', {}).get(f'{sentence[i-1]}:{rad4}', 0)
            rad3 = get_radical(sentence[i-1])
            if rad3:
                score += self.model.get('LSRID', {}).get(f'{rad3}:{sentence[i]}', 0)
            if rad3 and rad4:
                score += self.model.get('RAD', {}).get(f'{rad3}:{rad4}', 0)

            score += self.model.get('BW2', {}).get(sentence[i - 1:i + 1], 0)
            if i > 1:
              score += self.model.get('UW2', {}).get(sentence[i - 2], 0)
            score += self.model.get('UW3', {}).get(sentence[i - 1], 0)
            score += self.model.get('UW4', {}).get(sentence[i], 0)
            if i + 1 < len(sentence):
              score += self.model.get('UW5', {}).get(sentence[i + 1], 0)

            if score > 0:
                chunks.append(sentence[i])
            else:
                chunks[-1] += sentence[i]
        return chunks

import json
with open('model.json', encoding="utf-8") as f:
  model = json.load(f)
parser = AdaBoostSegmenter(model)
print("_".join(parser.predict("在香港實施「愛國者治港」的過程中，反對派人士被拘捕，獨立媒體停止運作，監察與匿名舉報現象日益增多。")))