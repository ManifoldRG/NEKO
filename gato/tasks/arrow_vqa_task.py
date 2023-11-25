from collections import defaultdict
from gato.tasks.task import Task, TaskTypeEnum

import torch
from torch import nn
import typing
import datasets
import random
from transformers import AutoTokenizer
import torchvision

to_tensor = torchvision.transforms.ToTensor()

class VqaTask(Task):
    def __init__(self, task_type: TaskTypeEnum, tokenizer_model:str, vqa_dataset: str):
        """
        task_type should be VQA
        vqa_dataset is the name of a 🤗 dataset
        """
        super().__init__(task_type)
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.dataset: datasets.DatasetDict = typing.cast(datasets.DatasetDict, datasets.load_dataset(vqa_dataset))

    def sample_batch(self, batch_size):
        self.dataset.shuffle()
        batch_dicts = [defaultdict(list)] * batch_size
        for key, values in self.dataset["train"][:batch_size].items():
            for i, value in enumerate(values):
                batch_dicts[i][key] = value
        for d in batch_dicts:
            # Unsqueezing to add a "batch" dimension because when gato_policy calls tokenize_input_dicts
            # and the forward gets called on the ImageEmbedding, it expects a (B, C, H, W) shape.
            # Alternatively, maybe we can revisit the return type of these `sample_batch` functions.
            d["images"] = to_tensor(d["image"].resize((256, 256))).unsqueeze(0)
        return batch_dicts

    def evaluate(self, model, num_examples_to_test=50, deterministic=True, log_examples_to_output=False):
        tokenizer = model.text_tokenizer
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0

        if num_examples_to_test > len(self.dataset['test']):
            print(f'num_examples_to_test chosen is more than test examples, so setting it to whole test dataset.')
            num_examples_to_test = len(self.dataset['test'])

        if log_examples_to_output:
            print(f'--- examples ---')
        for idx in range(num_examples_to_test):
            image = self.dataset['test'][idx]['image']
            question = self.dataset['test'][idx]['question']
            answer_idx = random.randint(0, len(self.dataset['test'][idx]['answers'])-1) # randomly choose an answer out of the set of answers
            target_answer = self.dataset['test'][idx]['answers'][answer_idx]['answer']
            target_tokens = tokenizer.encode(target_answer)

            # Generate prediction
            pred_logits, pred_answer = model.predict_answer(image, question, max_length=len(target_tokens), deterministic=deterministic)
            if log_examples_to_output and idx%10==0:
                print(f'Target answer: {target_answer} \n Predicted answer : {pred_answer}')
                print("----")

            # Calculate loss
            loss = loss_fn(pred_logits, torch.tensor(target_tokens).to(model.device))
            total_loss += loss.item()
            total_tokens += len(target_tokens)
        if log_examples_to_output:
            print(f'--- examples end ---')

        avg_loss = total_loss / num_examples_to_test
        perplexity = torch.exp(torch.tensor(avg_loss))

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity.item()
        }
        return metrics


# test code
if __name__ == '__main__':
    # replace the following directories and files names with your directories and files
    task = VqaTask(task_type                = TaskTypeEnum.VQA,
                   tokenizer_model          = 'gpt2',
                   vqa_dataset              = 'HuggingFaceM4/VQAv2')

    batch = task.sample_batch(5)
    print(batch)
