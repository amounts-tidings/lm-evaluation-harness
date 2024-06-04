"""
Evaluation tasks from the Country_city_animals dataset.
Structure of this file is modeled after the MMLU benchmark (hendrycks_test.py)
"""

from lm_eval.base import MultipleChoiceTask, Task, rf
import numpy as np
import evaluate

names = ['Eval_QA', 'Eval_multiple_choice', 'Eval_reverse', 'Eval_indirect_reasoning', 'Eval_animal_commonsense', 'Eval_2hop_reasoning', 'Eval_2hop_reasoning_raw']


def create_all_tasks():
    return {
        name: create_task("xiaozeroone/Country-city-animals", name) for name in names
    }

def create_task(dataset_name, subset):
    if subset in ['Eval_multiple_choice', 'Eval_2hop_reasoning']:
        class MC(GeneralMC):
            def __init__(self):
                super().__init__(dataset_name, subset)

        return MC
    else:
        class QA(GeneralQA):
            def __init__(self):
                super().__init__(dataset_name, subset)
    
        return QA


class GeneralMC(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, dataset_name, subset):
        self.DATASET_PATH = dataset_name
        self.DATASET_NAME = subset
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        kwargs["description"] = f"The following are multiple choice questions (with answers)."
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):

        def format_example(doc, keys):
            """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """

            question = doc["question"].strip()
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt = f"{question}\n{choices}Answer:"
            return prompt
            
        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]


squad_metric = evaluate.load("squad")

def _squad_metric(predictions, references):
    return squad_metric.compute(predictions=predictions, references=references)


class GeneralQA(Task):
    VERSION = 1
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, dataset_path, dataset_name):
        self.DATASET_PATH = dataset_path
        self.DATASET_NAME = dataset_name
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return (
            "Question: "
            + doc["question"]
            # + "\n\n"
            + " "
            + "Answer:"
        )

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        answer = doc['answer']
        return " " + answer

    def construct_requests(self, doc, ctx):

        continuation = rf.greedy_until(ctx, {"until": ["\n"], "max_length": 50})
        return continuation

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        try:
            self.fake_id = self.fake_id + 1
        except AttributeError:
            self.fake_id = 0

        continuation = results[0]

        predictions = {
            "id": str(self.fake_id),
            "prediction_text": continuation,
        }

        references = {
            "id": str(self.fake_id),
            "answers": [{'text': doc["answer"], 'answer_start': 0}],  # construct squad-like format so that we can use squad metric
        }

        metrics = _squad_metric(predictions=[predictions], references=[references])

        return {
            "exact": (
                predictions,
                references,
                metrics["exact_match"],
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
                metrics["f1"],
            ),  # The F-score of predicted tokens versus the gold answer
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "exact": lambda items: np.mean([item[2] for item in items]),
            "f1": lambda items: np.mean([item[2] for item in items]),
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)