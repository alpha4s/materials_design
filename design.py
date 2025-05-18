# design.py
import torch
from skopt import gp_minimize
from skopt.space import Real

def gradient_design(pred, decoder, targets, lr, steps):
    z = torch.randn(1, decoder.input_dim - targets.shape[-1], requires_grad=True, device=decoder.weight.device)
    opt = torch.optim.Adam([z], lr=lr)
    tgt = torch.tensor(targets, device=z.device)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(pred(z), tgt)
        loss.backward(); opt.step()
    return decoder(torch.cat([z, tgt.unsqueeze(0)],1)).cpu().detach().numpy()

def bo_design(pred, decoder, targets, n_calls=50):
    def obj(z):
        zt = torch.tensor(z, device=next(pred.parameters()).device).unsqueeze(0)
        return float(((pred(zt).cpu().numpy().ravel() - targets)**2).mean())
    space = [Real(-3,3)] * decoder.in_features
    res = gp_minimize(obj, space, n_calls=n_calls)
    z_opt = torch.tensor(res.x, device=next(pred.parameters()).device).unsqueeze(0)
    return decoder(torch.cat([z_opt, torch.tensor(targets, device=z_opt.device).unsqueeze(0)],1)).cpu().numpy()
