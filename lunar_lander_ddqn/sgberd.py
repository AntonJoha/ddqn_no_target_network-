import torch
from torch.optim.optimizer import Optimizer, required


def SGBerD_wrapper(params, lr, config=None):
    if config is None:
        config = {}

    prob = config.get("prob", 0.5)
    magnitude = config.get("magnitude", 1.0)
    noise_decay = config.get("noise_decay", False)

    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    if prob <= 0 or prob >= 1:
        raise ValueError("probability must be in (0, 1)")

    return SGBerD(params, lr=lr, prob=prob, magnitude=magnitude, noise_decay=noise_decay)


class SGBerD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        prob=0.5,
        magnitude=1.0,
        noise_decay=False,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            prob=prob,
            magnitude=magnitude,
            noise_decay=noise_decay,
        )
        super().__init__(params, defaults)

        self.prob = float(prob)
        self.offset = -self.prob
        self.magnitude = float(magnitude)
        self.noise_rate = float(lr if not noise_decay else noise_decay)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if d_p.is_sparse:
                    raise RuntimeError("SGBerD does not support sparse gradients")

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = d_p.clone().detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)

                noise_lr = self.magnitude * self.noise_rate
                if noise_lr != 0:
                    bern = torch.bernoulli(torch.full_like(p, self.prob))
                    noise = (bern + self.offset) * noise_lr
                    p.add_(noise)

        return loss
