import torch
from torch.utils.data import DataLoader
from gato.tasks.task import Task

class NextTokenPredictionTask(Task):
    def __init__(self, dataset, tokenizer, batch_size=32, device='cpu'):
        super().__init__()
        self.dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.tokenizer = tokenizer
        self.device = device

    def sample_batch(self, vanilla_batch_size, prompted_batch_size, device, max_tokens=1024):
        # Since we are doing next token prediction, we don't need a prompted batch size
        # We will just use the vanilla batch size
        for i, (input_tokens, target_tokens) in enumerate(self.dataset):
            if i == vanilla_batch_size:
                break
            yield {
                'input_tokens': input_tokens.to(device),
                'target_tokens': target_tokens.to(device)
            }

    def evaluate(self, model, n_iterations):
        # Evaluation logic for next token prediction
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        loss_function = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for i, batch in enumerate(self.sample_batch(n_iterations, 0, self.device)):
                input_tokens = batch['input_tokens']
                target_tokens = batch['target_tokens']
                predictions = model(input_tokens)
                loss = loss_function(predictions.view(-1, predictions.size(-1)), target_tokens.view(-1))
                total_loss += loss.item()

                predicted_tokens = predictions.argmax(dim=-1)
                total_correct += (predicted_tokens == target_tokens).sum().item()
                total_tokens += target_tokens.numel()

        return {
            'mean_loss': total_loss / n_iterations,
            'accuracy': total_correct / total_tokens
        }
