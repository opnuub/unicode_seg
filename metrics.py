from typing import List
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score

class Metrics(ABC):
    def __init__(self, y_test: List[str], y_pred: List[str]):
        self.actual = [label for word in y_test for label in ([0] * (len(word) - 1)) + [1]]
        self.pred = [label for word in y_pred for label in ([0] * (len(word) - 1)) + [1]]

    @property
    @abstractmethod
    def accuracy(self) -> float:
        pass

    @abstractmethod
    def get_f1_score(self, is_weighted: bool = False) -> float:
        pass


# Inheritance
class BinaryMetrics(Metrics):
    def __init__(self, y_test: List[str], y_pred: List[str]):
        super().__init__(y_test, y_pred)
        self._accuracy = accuracy_score(self.actual, self.pred)
        self._f1_score = f1_score(self.actual, self.pred)

    @property
    def accuracy(self) -> float:
        return self._accuracy
    
    def get_f1_score(self, is_weighted: bool = False) -> float:
        return self._f1_score

class MultiClassMetrics(Metrics):
    def __init__(self, y_test: List[str], y_pred: List[str], classes: List[str]):
        super().__init__(y_test, y_pred)
        self.classes = classes # e.g. ['b', 'i', 'e', 's']
        # Only implemented bies
        if self.classes == 'bies':
            self.actual = self.convert_to_bies(self.actual)
            self.pred = self.convert_to_bies(self.pred)

            self._accuracy = accuracy_score(self.actual, self.pred)
            self._f1_score_macro = f1_score(self.actual, self.pred, average='macro')
            self._f1_score_weighted = f1_score(self.actual, self.pred, average='weighted')

    @property
    def accuracy(self) -> float:
        return self._accuracy
    
    def get_f1_score(self, is_weighted: bool = False) -> float:
        if is_weighted:
            return self._f1_score_weighted
        else:
            return self._f1_score_macro

    def convert_to_bies(self, arr):
        if len(arr) == 1: return ['s']

        arr[0] = 'b' if arr[0] == 0 else 's'
        for i in range(1, len(arr)):
            # right after break
            if arr[i-1] == 's' or arr[i-1] == 'e':
                if arr[i]: 
                    arr[i] = 's'
                else: 
                    arr[i] = 'b'
            # previously b or i
            else:
                if arr[i]:
                    arr[i] = 'e'
                else:
                    arr[i] = 'i'
        return arr
        