"""Prediction-only boundary extrapolation hypotheses for a past-trained flow."""
import math
import torch


class BoundaryFlow:
    def __init__(self, net, policy):
        if policy not in ['original', 'timeclamp', 'ballistic']:
            raise ValueError('Undeclared boundary policy')
        self.net = net
        self.policy = policy
        self.cutoff, self.origin = net.cutoff, net.origin

    def encode(self, values):
        return self.net.encode(values)

    def trajectory(self, z, times, step=.125):
        if self.policy == 'original':
            return self.net.trajectory(z, times, step=step)
        if step <= 0 or len(times) < 2 or float(times[0]) != 0.:
            raise ValueError('Invalid forecast integration')
        if any(float(b) < float(a) for a, b in zip(times[:-1], times[1:])):
            raise ValueError('Future forecast times must be nondecreasing')
        boundary = self.cutoff - self.origin
        initial = z
        initial_velocity = self.net.velocity(boundary, initial)
        history = [z]
        for previous, target in zip(times[:-1], times[1:]):
            if self.policy == 'ballistic':
                z = initial + float(target) * initial_velocity
            else:
                span = float(target - previous)
                count = max(1, int(math.ceil(span / step)))
                dt = span / count
                for _ in range(count):
                    a = self.net.velocity(boundary, z)
                    b = self.net.velocity(boundary, z + dt * a / 2)
                    c = self.net.velocity(boundary, z + dt * b / 2)
                    d = self.net.velocity(boundary, z + dt * c)
                    z = z + dt * (a + 2*b + 2*c + d) / 6
            history.append(z)
        return torch.stack(history)
