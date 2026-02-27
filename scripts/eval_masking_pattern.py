"""Quick eval of MaskingPatternEval on a pretrained SPD model.

Usage:
    python scripts/eval_masking_pattern.py
"""

import torch

from spd.experiments.tms.bss_models import PingPongModel
from spd.metrics.masking_pattern_eval import MaskingPatternEval
from spd.models.component_model import ComponentModel

MODEL_PATH = "wandb:paramluhadiya/spd/runs/s-898e5c08"
DEVICE = "cuda"
N_BATCHES = 20
BATCH_SIZE = 256


@torch.no_grad()
def main() -> None:
    model = ComponentModel.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(DEVICE)

    assert isinstance(model.target_model, PingPongModel)
    D = model.target_model.D
    d = model.target_model.d
    num_blocks = model.target_model.num_blocks

    metric = MaskingPatternEval(
        model=model,
        device=DEVICE,
        D=D,
        d=d,
        cos_sim_threshold=0.75,
        ci_threshold=0.1,
    )

    from spd.experiments.tms.pingpong_decomposition import PingPongDataset

    dataset = PingPongDataset(D=D, d=d, num_blocks=num_blocks, device=DEVICE)

    for i in range(N_BATCHES):
        batch = dataset.generate_batch(BATCH_SIZE)
        out = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            sampling="binomial",
            detach_inputs=True,
        )
        metric.update(batch=batch, ci=ci)
        print(f"  batch {i + 1}/{N_BATCHES}")

    results = metric.compute()
    print("\n=== MaskingPatternEval Results ===")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
