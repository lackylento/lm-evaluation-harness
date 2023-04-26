
import lm_eval.datasets.coqa.coqa
from lm_eval.base import MultipleChoiceTask
import numpy as np


class CHEF(Task):
    VERSION=0
    DATASET_PATH = "/users10/mzsun/evaluation/lm_eval/datasets/CHEF"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        return {
            "claim": doc['claim'],  # The claim prompt.
            "evidence": doc["evidence"],  # The list of choices.
            "label": doc["label"],  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return "通过以下{}判断{}".fomat(doc["evidence"],doc["claim"])
    
    def doc_to_target(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["label"]