from typing import Dict, List, Tuple

import torch
from neptune import Run


def get_device():
    return torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')


def logged_weighted_average(run: Run):
    def weighted_average(metrics: List[Tuple[float, int, Dict[str, float]]]) -> Tuple[float, int, Dict[str, float]]:
        print(metrics)
        total_accuracy = sum(
            [metric[1]["accuracy"]*metric[0] for metric in metrics])
        total_loss = sum(
            [metric[1]["loss"]*metric[0] for metric in metrics]
        )
        for metric in metrics:
            run[f"test/accuracy/client_{metric[1]['client_id']}"].append(metric[1]["accuracy"])
            run[f"test/loss/client_{metric[1]['client_id']}"].append(metric[1]["loss"])
        num_examples = sum([metric[0] for metric in metrics])
        run["test/accuracy/mean"].append(total_accuracy/num_examples)
        run["test/loss/mean"].append(total_loss/num_examples)
        return {"num_examples": num_examples, "accuracy": total_accuracy/num_examples}
    return weighted_average
