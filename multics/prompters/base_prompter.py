from ..models.model_manager import ModelManager
import torch


def tokenize_long_prompt(tokenizer, prompt, max_length=None):
    length = tokenizer.model_max_length if max_length is None else max_length
    tokenizer.model_max_length = 99999999
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    max_length = (input_ids.shape[1] + length - 1) // length * length
    tokenizer.model_max_length = length
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids
    num_sentence = input_ids.shape[1] // length
    input_ids = input_ids.reshape((num_sentence, length))
    return input_ids


class BasePrompter:
    def __init__(self):
        self.refiners = []
        self.extenders = []

    def load_prompt_refiners(self, model_manager: ModelManager, refiner_classes=[]):
        for refiner_class in refiner_classes:
            refiner = refiner_class.from_model_manager(model_manager)
            self.refiners.append(refiner)

    def load_prompt_extenders(self, model_manager: ModelManager, extender_classes=[]):
        for extender_class in extender_classes:
            extender = extender_class.from_model_manager(model_manager)
            self.extenders.append(extender)

    @torch.no_grad()
    def process_prompt(self, prompt, positive=True):
        if isinstance(prompt, list):
            prompt = [
                self.process_prompt(prompt_, positive=positive) for prompt_ in prompt
            ]
        else:
            for refiner in self.refiners:
                prompt = refiner(prompt, positive=positive)
        return prompt

    @torch.no_grad()
    def extend_prompt(self, prompt: str, positive=True):
        extended_prompt = dict(prompt=prompt)
        for extender in self.extenders:
            extended_prompt = extender(extended_prompt)
        return extended_prompt
