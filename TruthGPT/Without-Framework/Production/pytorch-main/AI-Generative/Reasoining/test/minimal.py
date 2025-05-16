import os
import re
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file, load_file

# --- Kalman Filter con memoria exponencial y momentum ---

class KalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float, memory_size: int = 1000):
        self.Q = process_noise
        self.R = measurement_noise
        self.mu = 0.0
        self.P = 1.0
        self.memory = []
        self.memory_size = memory_size
        self.momentum = 0.9
        self.velocity = 0.0

    def update(self, measurement: float) -> float:
        mu_pred = self.mu + self.momentum * self.velocity
        P_pred = self.P + self.Q

        K = P_pred / (P_pred + self.R)
        innovation = measurement - mu_pred
        self.mu = mu_pred + K * innovation
        self.P = (1 - K) * P_pred + self.Q

        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * innovation

        self.memory.append(measurement)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        return self.mu

    def get_statistics(self) -> Tuple[float, float]:
        if not self.memory:
            return 0.0, 1.0

        weights = np.exp(np.linspace(-1, 0, len(self.memory)))
        weights /= weights.sum()

        weighted_mean = np.average(self.memory, weights=weights)
        weighted_std = np.sqrt(np.average((np.array(self.memory) - weighted_mean) ** 2, weights=weights))

        return weighted_mean, weighted_std

# --- Mixture-of-Experts para Kalman Filters ---

class ExpertKalmanMoE:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.experts = defaultdict(lambda: KalmanFilter(process_noise, measurement_noise))

    def update(self, task: str, reward: float) -> float:
        return self.experts[task].update(reward)

    def get_statistics(self, task: str) -> Tuple[float, float]:
        return self.experts[task].get_statistics()

# --- Función de recompensa ejemplo (IoU) ---

def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union if union != 0 else 0.0

    def process_answer(content, task_type):
        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not match:
            return [0, 0, 0, 0] if task_type == 'detection' else ""
        content_answer = match.group(1).strip()
        if task_type == 'detection':
            bbox_match = re.search(r'\[(\s*-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]', content_answer)
            return [int(bbox_match.group(i)) for i in range(1, 5)] if bbox_match else [0, 0, 0, 0]
        return content_answer

    def process_solution(content, task_type):
        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not match:
            return [0, 0, 0, 0] if task_type == 'detection' else ""
        sol_answer = match.group(1).strip()
        if task_type == 'detection':
            bbox_match = re.search(r'\[(\s*-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]', sol_answer)
            return [int(bbox_match.group(i)) for i in range(1, 5)] if bbox_match else [0, 0, 0, 0]
        return sol_answer

    contents = [completion[0]["content"] for completion in completions]
    task_list = kwargs.get('task')
    problem_list = kwargs.get('problem')

    moe = ExpertKalmanMoE()
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, task, _problem in zip(contents, solution, task_list, problem_list):
        reward = 0.0
        try:
            if task == 'detection':
                bbox = process_answer(content, task)
                gt = process_solution(sol, task)
                if iou(bbox, gt) > 0.5:
                    reward = 1.0
            elif task == 'classify':
                student = process_answer(content, task)
                gt = process_solution(sol, task)
                if gt in student:
                    reward = 1.0
            elif task == 'math':
                # Aquí deberías añadir parse y verify si los usas realmente
                gt = process_solution(sol, task)
                student = process_answer(content, task)
                if gt in student:
                    reward = 1.0
            elif task == 'coco_vqa':
                # Aquí asumes que gte_model y cos_sim están definidos
                gt = process_solution(sol, task)
                student = process_answer(content, task)
                embeddings = gte_model.encode([student, gt])
                sim_cosine = float(cos_sim(embeddings[0], embeddings[1]).cpu().numpy()[0])
                reward = 1.0 if sim_cosine > 0.95 else 0.2 if sim_cosine > 0.9 else 0.0
        except:
            reward = 0.0

        smoothed = moe.update(task, reward)
        rewards.append(smoothed)

        if os.getenv("DEBUG_MODE") == "true":
            with open(os.getenv("LOG_PATH"), "a") as f:
                f.write(f"--- {current_time} | Task: {task} | Reward: {reward:.2f} | Smoothed: {smoothed:.2f} ---\n")
                f.write(f"Problem: {_problem}\nContent: {content}\nSolution: {sol}\n")

    return rewards

def format_reward(completions, **kwargs):
    tasklist = kwargs.get('task')
    matches = []
    for task, completion in zip(tasklist, completions):
        content = completion[0]["content"]
        if task == 'detection':
            pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        else:
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        matches.append(re.fullmatch(pattern, content, re.DOTALL))
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": iou_reward,
    "format": format_reward,
}

# --- Modelo Transformer simple ---

class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, max_seq_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        emb = self.embed(x)  # (B, S, E)
        emb = emb.permute(1, 0, 2)  # (S, B, E)
        out = self.transformer_encoder(emb)
        out = out.permute(1, 0, 2)  # (B, S, E)
        logits = self.fc_out(out)
        return logits

def save_model_safetensors(model, path):
    state_dict = model.state_dict()
    save_file(state_dict, path)

def load_model_safetensors(model_class, path):
    model = model_class()
    state_dict = load_file(path)
    model.load_state_dict(state_dict)
    return model

def train_dummy(model, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        inputs = torch.randint(0, 10000, (8, 20))
        targets = torch.randint(0, 10000, (8, 20))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view
