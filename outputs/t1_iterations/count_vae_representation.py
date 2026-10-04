"""Own matched observation-model VAE probe, not an scVI reproduction."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ObservationVAE(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ('count_nb', 'log_gaussian'):
            raise ValueError('Unknown observation model')
        self.kind = kind
        self.encoder = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 128))
        self.observation_scale = torch.nn.Parameter(torch.zeros(128))

    def encode(self, values):
        mean, logvar = self.encoder(values).chunk(2, dim=-1)
        return mean, logvar.clamp(-8., 8.)

    def decode_fractions(self, z):
        logits = self.decoder(z)
        # Background category accounts for genes outside the fixed128 subset.
        logits = torch.cat([logits, torch.zeros_like(logits[:, :1])], dim=1)
        return torch.softmax(logits, dim=1)[:, :128]

    def loss(self, counts, libraries, epsilon):
        observed = torch.log1p(10000. * counts / libraries[:, None])
        mean, logvar = self.encode(observed)
        z = mean + torch.exp(.5 * logvar) * epsilon
        fractions = self.decode_fractions(z).clamp_min(1e-9)
        scale = torch.nn.functional.softplus(self.observation_scale) + .01
        if self.kind == 'count_nb':
            mu = libraries[:, None] * fractions
            logp = torch.lgamma(counts + scale) - torch.lgamma(scale) - torch.lgamma(counts + 1)
            logp = logp + scale * (torch.log(scale) - torch.log(scale + mu))
            logp = logp + counts * (torch.log(mu) - torch.log(scale + mu))
            reconstruction = -logp.mean()
        else:
            predicted = torch.log1p(10000. * fractions)
            reconstruction = (.5 * ((observed - predicted) / scale).square() + torch.log(scale)).mean()
        kl = .5 * (mean.square() + logvar.exp() - logvar - 1.).sum(1).mean()
        return reconstruction + .01 * kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='count_vae_representation_preflight')
    args = parser.parse_args()
    entry = json.loads((HERE / 'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments'][args.experiment]
    run, public = (HERE / entry['run']).resolve(), (HERE / entry['report']).resolve()
    if not run.is_relative_to(HERE / 'private') or not public.is_relative_to(HERE):
        raise ValueError('Evidence outside project')
    if run.exists() or public.exists():
        raise ValueError('Never duplicate preflight')
    run.mkdir(parents=True)
    report = {'status': 'running', 'new_scoring_batch': False, 'score': None, 'reward': 0}
    public.write_text(json.dumps(report, indent=2))
    try:
        context = HERE / 'private/hidden_protein_count_bridge_repair_01/past_counts_context.npz'
        expected = '49ac69cc7d42c6c8238d704d3e4b4c1207a458d71081aa110e40474e28175956'
        if digest(context) != expected:
            raise ValueError('Frozen actual count context changed')
        source_sha = digest(__file__)
        plan = {'context_sha256': expected, 'source_sha256': source_sha, 'fit_max_stage': 8.25,
                'rank': 8, 'width': 64, 'genes': 128, 'batch': 64, 'steps_per_arm': 3,
                'matched_control': 'Same architecture,128observation parameters, pairedbatch/epsilon; NB counts vs Gaussian lognormalized likelihood, same KL.01.',
                'scope': 'Resource/finite-gradient and representation encode/decode checks only; not scVI author implementation or predictive validation.'}
        (run / 'plan.json').write_text(json.dumps(plan, indent=2))
        with np.load(context) as data:
            counts, libraries, stages = data['counts'], data['libraries'], data['stages']
        if sorted(np.unique(stages)) != [7.5, 7.75, 8., 8.25]:
            raise ValueError('Past-only support changed')
        if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128':
            raise ValueError('Pinned CUDA required; no CPU fallback')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.use_deterministic_algorithms(True)
        results = {}
        for kind in ['log_gaussian', 'count_nb']:
            torch.manual_seed(20261004)
            torch.cuda.reset_peak_memory_stats()
            model = ObservationVAE(kind).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            rng = torch.Generator().manual_seed(20261004)
            losses = []
            started = time.perf_counter()
            for _ in range(3):
                rows = torch.randint(len(counts), (64,), generator=rng).numpy()
                x = torch.tensor(counts[rows], device='cuda')
                exposure = torch.tensor(libraries[rows], dtype=torch.float32, device='cuda')
                epsilon = torch.randn(64, 8, generator=rng).cuda()
                optimizer.zero_grad()
                loss = model.loss(x, exposure, epsilon)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite VAE objective')
                loss.backward()
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise ValueError('Nonfinite/missing VAE gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                values = torch.log1p(10000. * x / exposure[:, None])
                mean, _ = model.encode(values)
                fractions = model.decode_fractions(mean)
                if not torch.isfinite(mean).all() or not torch.isfinite(fractions).all() or (fractions < 0).any() or (fractions.sum(1) > 1.).any():
                    raise ValueError('Invalid representation/decoder fractions')
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if peak > 4.5 * 1024 ** 3:
                raise ValueError('GPU memory cap exceeded')
            results[kind] = {'parameters': sum(v.numel() for v in model.parameters()), 'losses': losses,
                             'seconds': time.perf_counter() - started, 'gpu_peak_bytes': peak}
            model.cpu().eval()
            probe = values.cpu()
            checkpoint = run / (kind + '_preflight.pt')
            torch.save(model.state_dict(), checkpoint)
            clone = ObservationVAE(kind).eval()
            clone.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(model.encode(probe)[0], clone.encode(probe)[0], rtol=0, atol=0)
            results[kind]['checkpoint_replay_exact'] = True
            del optimizer, model, clone
        if digest(context) != expected or digest(__file__) != source_sha:
            raise ValueError('Source/input changed after preflight')
        report.update(status='completed', arms=results, past_cells=len(counts), genes=128,
                      scope=plan['scope'], next_requirement='Freeze paired representation/temporal objective and count-to-original-encoder bridge before scoring; neither3step model is trained for prediction.')
    except Exception as exc:
        (run / 'traceback.txt').write_text(traceback.format_exc())
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    public.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
