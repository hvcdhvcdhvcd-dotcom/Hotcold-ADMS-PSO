"""ADMS-PSO optimiser for the HOTCOLD wearable-patch representation.

Positions are stored in [0, 1] x [0, 1] relative to a pedestrian bounding
box. This is the coordinate system used by ``ParticleToPatch`` and makes a
solution transferable between differently sized people and images.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import gamma, pi, sin
from typing import List
import torch


@dataclass
class ADMSPSOConfig:
    num_patches: int = 4
    grid_size: int = 3
    cell_size: int = 16
    swarm_size: int = 30
    iterations: int = 100
    lambda_size: float = 0.05
    lambda_overlap: float = 0.10
    lambda_valid: float = 1.0
    w_max: float = 0.9
    w_min: float = 0.4
    c1_max: float = 2.5
    c1_min: float = 0.5
    c2_min: float = 0.5
    c2_max: float = 2.5
    c3: float = 1.0
    distance_scale: float = 0.2
    archive_size: int = 5
    diversity_threshold: float = 0.3
    shape_diversity_weight: float = 0.7
    mutation_max: float = 0.10
    mutation_min: float = 0.01
    overlap_threshold: float = 0.3
    cooperative_interval: int = 5
    stagnation_iterations: int = 10
    levy_beta: float = 1.5
    levy_alpha: float = 20.0
    levy_fraction: float = 0.30
    seed: int | None = None


class SwarmParameters:
    """Compatibility result consumed by val_patch.py."""
    pass


class OptimizeFunction:
    def __init__(self, detector, patch_size, device, config=None):
        from utils_patch import PatchApplier
        from ptop import ParticleToPatch
        self.detector, self.device = detector, device
        self.config = config or ADMSPSOConfig()
        self.ptp = ParticleToPatch(patch_size, cell_size=self.config.cell_size)
        self.pa, self.patch_size = PatchApplier(), patch_size

    def set_para(self, targets, imgs):
        self.targets, self.imgs = targets, imgs

    def evaluate_detector(self, position):
        """Preserve the repository's original detector objective exactly."""
        with torch.no_grad():
            patch, mask = self.ptp(position, self.targets, self.imgs)
            output, _ = self.detector(self.pa(self.imgs, patch, mask))
            return output[:, :, 4].max(dim=1).values.mean()

    def evaluate(self, position):
        detector_loss = self.evaluate_detector(position)
        shapes, locations = position
        h, w = self.imgs.shape[-2:]
        # n*L^2/9 is paper-area in pixels. Image-area normalisation keeps it
        # comparable with the detector confidence objective.
        size_loss = shapes.float().sum() * self.config.cell_size ** 2 / 9.0 / float(h * w)
        return detector_loss + self.config.lambda_size * size_loss + self.config.lambda_overlap * _overlap_penalty(locations, self.patch_size, self.config.overlap_threshold) + self.config.lambda_valid * _valid_penalty(locations)


def _clone(position):
    return [position[0].clone(), position[1].clone()]


def _valid_penalty(locations):
    return torch.relu(-locations).sum() + torch.relu(locations - 1).sum()


def _overlap_penalty(locations, side, threshold):
    total = locations.new_zeros(())
    for first in range(len(locations)):
        for second in range(first + 1, len(locations)):
            dx, dy = (locations[first] - locations[second]).abs()
            inter = torch.relu(locations.new_tensor(side) - dx) * torch.relu(locations.new_tensor(side) - dy)
            total = total + torch.relu(inter / (2 * side * side - inter).clamp_min(1e-8) - threshold)
    return total


class Particle:
    def __init__(self, config, device):
        self.config, self.device = config, device
        shapes = (torch.rand(config.num_patches, config.grid_size, config.grid_size, device=device) < 0.5).float()
        self.position = [self.repair(shapes), torch.rand(config.num_patches, 2, device=device)]
        self.position_velocity = torch.empty(config.num_patches, 2, device=device).uniform_(-.2, .2)
        self.shape_velocity = torch.empty_like(shapes).uniform_(-1, 1)
        self.pbest_position, self.pbest_value = _clone(self.position), torch.tensor(float('inf'), device=device)

    def repair(self, shapes):
        for index in torch.where(shapes.flatten(1).sum(1) == 0)[0].tolist():
            shapes[index, torch.randint(shapes.shape[1], (), device=shapes.device), torch.randint(shapes.shape[2], (), device=shapes.device)] = 1
        return shapes


class PSO:
    def __init__(self, swarm_size, device, config=None):
        self.config = config or ADMSPSOConfig(swarm_size=swarm_size)
        self.device = device
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.swarm = [Particle(self.config, device) for _ in range(self.config.swarm_size)]
        self.gbest_position, self.gbest_particle = None, None
        self.gbest_value = torch.tensor(float('inf'), device=device)
        self.archive: List[dict] = []

    def optimize(self, function):
        self.fitness_function = function

    def distance(self, first, second):
        return torch.linalg.vector_norm(first - second, dim=1).mean() / 2 ** .5

    def diversity(self, first, second):
        return self.config.shape_diversity_weight * (first[0] != second[0]).float().mean() + (1 - self.config.shape_diversity_weight) * self.distance(first[1], second[1])

    def update_archive(self, candidates):
        for candidate in sorted(candidates, key=lambda x: float(x['value'])):
            if all(float(self.diversity(candidate['position'], entry['position'])) >= self.config.diversity_threshold for entry in self.archive):
                self.archive.append({'position': _clone(candidate['position']), 'value': candidate['value'].detach().clone()})
        self.archive.sort(key=lambda x: float(x['value']))
        self.archive = self.archive[:self.config.archive_size]
        if not self.archive:
            candidate = min(candidates, key=lambda x: float(x['value']))
            self.archive.append({'position': _clone(candidate['position']), 'value': candidate['value'].detach().clone()})

    def archive_memory(self, locations):
        entry = min(self.archive, key=lambda x: float(self.distance(locations, x['position'][1])))
        d = self.distance(locations, entry['position'][1])
        return entry['position'][0], torch.exp(-(d ** 2) / (2 * self.config.distance_scale ** 2))

    def evaluate_swarm(self):
        values = []
        for particle in self.swarm:
            value = self.fitness_function.evaluate(particle.position)
            values.append(value)
            if value < particle.pbest_value:
                particle.pbest_value, particle.pbest_position = value.detach().clone(), _clone(particle.position)
            if value < self.gbest_value:
                self.gbest_value, self.gbest_position, self.gbest_particle = value.detach().clone(), _clone(particle.position), deepcopy(particle)
        return values

    def cooperate(self, particle):
        # The supplied pseudocode's candidate equals the current point. Use a
        # bounded local proposal so cooperative optimisation actually searches.
        current = self.fitness_function.evaluate(particle.position)
        for patch in range(self.config.num_patches):
            proposal = _clone(particle.position)
            proposal[1][patch] = (proposal[1][patch] + torch.empty(2, device=self.device).uniform_(-.08, .08)).clamp(0, 1)
            value = self.fitness_function.evaluate(proposal)
            if value < current:
                particle.position, current = proposal, value

    def levy_step(self):
        beta = self.config.levy_beta
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        return torch.randn(2, device=self.device) * sigma / torch.randn(2, device=self.device).abs().clamp_min(1e-8).pow(1 / beta)

    def recover(self, values):
        worst = torch.topk(torch.stack([v.detach() for v in values]), max(1, round(self.config.levy_fraction * len(self.swarm)))).indices.tolist()
        for index in worst:
            particle = self.swarm[index]
            for patch in range(self.config.num_patches):
                candidate = _clone(particle.position)
                candidate[1][patch] = (candidate[1][patch] + self.config.levy_alpha / 640 * self.levy_step()).clamp(0, 1)
                opposite = _clone(candidate)
                opposite[1][patch] = 1 - candidate[1][patch]
                particle.position = candidate if self.fitness_function.evaluate(candidate) < self.fitness_function.evaluate(opposite) else opposite

    def run(self):
        if not hasattr(self, 'fitness_function'):
            raise RuntimeError('Call optimize(function) before run().')
        values = self.evaluate_swarm()
        self.update_archive([{'position': p.position, 'value': v} for p, v in zip(self.swarm, values)])
        stagnant = 0
        for iteration in range(1, self.config.iterations + 1):
            ratio = iteration / self.config.iterations
            w = self.config.w_max - (self.config.w_max - self.config.w_min) * ratio
            c1 = (1 - ratio) * self.config.c1_max + ratio * self.config.c1_min
            c2 = (1 - ratio) * self.config.c2_min + ratio * self.config.c2_max
            mutation = self.config.mutation_max - (self.config.mutation_max - self.config.mutation_min) * ratio
            old_best = self.gbest_value.clone()
            for particle in self.swarm:
                particle.position_velocity = w * particle.position_velocity + c1 * torch.rand_like(particle.position_velocity) * (particle.pbest_position[1] - particle.position[1]) + c2 * torch.rand_like(particle.position_velocity) * (self.gbest_position[1] - particle.position[1])
                particle.position[1] = (particle.position[1] + particle.position_velocity).clamp(0, 1)
                sp = torch.exp(-(self.distance(particle.position[1], particle.pbest_position[1]) ** 2) / (2 * self.config.distance_scale ** 2))
                sg = torch.exp(-(self.distance(particle.position[1], self.gbest_position[1]) ** 2) / (2 * self.config.distance_scale ** 2))
                memory, sa = self.archive_memory(particle.position[1])
                particle.shape_velocity = w * particle.shape_velocity + c1 * torch.rand_like(particle.shape_velocity) * sp * (particle.pbest_position[0] - particle.position[0]) + c2 * torch.rand_like(particle.shape_velocity) * sg * (self.gbest_position[0] - particle.position[0]) + self.config.c3 * torch.rand_like(particle.shape_velocity) * sa * (memory - particle.position[0])
                particle.position[0] = (torch.rand_like(particle.shape_velocity) < torch.sigmoid(particle.shape_velocity)).float()
                particle.position[0] = particle.repair(torch.where(torch.rand_like(particle.position[0]) < mutation, 1 - particle.position[0], particle.position[0]))
            if iteration % self.config.cooperative_interval == 0:
                for particle in self.swarm:
                    self.cooperate(particle)
            values = self.evaluate_swarm()
            self.update_archive([{'position': p.position, 'value': v} for p, v in zip(self.swarm, values)])
            stagnant = stagnant + 1 if self.gbest_value >= old_best else 0
            if stagnant >= self.config.stagnation_iterations:
                self.recover(values)
                values = self.evaluate_swarm()
                self.update_archive([{'position': p.position, 'value': v} for p, v in zip(self.swarm, values)])
                stagnant = 0
        result = SwarmParameters()
        result.gbest_position, result.gbest_value = _clone(self.gbest_position), self.gbest_value.item()
        result.c1, result.c2, result.archive_size = c1, c2, len(self.archive)
        return result
