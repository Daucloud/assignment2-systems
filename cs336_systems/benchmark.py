import numpy as np
from rich.progress import Progress
from rich.panel import Panel
from rich.console import Console
from rich.traceback import install
import timeit
import torch
import click
from cs336_basics import model, optimizer, data, nn_utils
from utils import set_seed

console=Console()
install(show_locals=True)

def get_random_data(seq: int, vocab_size: int):
    return np.random.randint(low=0, high=vocab_size, size=(seq,))

def get_model(context_length, d_model, d_ff,  num_layers, num_heads, vocab_size=10000, rope_theta=10000.0):
    return model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

def train_steps(steps, lr_max, lr_min, Tw, Tc, model, opt, train_data, bs, clen, device, is_warmup=False):
    with Progress() as progress:
        base_desc="Warm up..." if is_warmup else "Training..."
        task=progress.add_task(base_desc, total=steps)
        for t in range(1,steps+1):
            lr=optimizer.get_cosine_lr(t, lr_max, lr_min, Tw, Tc)
            for param_group in opt.param_groups:
                param_group['lr']=lr
            opt.zero_grad()
            inputs, targets=data.get_batch(train_data, bs, clen, device)
            loss=nn_utils.cross_entropy(model(inputs), targets)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            progress.update(
                task,
                advance=1,
                description=f"{base_desc}Loss: {loss.item(): .4f}",
            )

@click.command()
@click.option('--context_length', type=int, default=1024)
@click.option('--d_model', type=int, default=768)
@click.option('--d_ff', type=int, default=3072)
@click.option('--num_layers', type=int, default=12)
@click.option('--num_heads', type=int, default=12)
@click.option('--warmup_steps', type=int, default=10)
@click.option('--steps', type=int, default=100)
@click.option('--lr_max', type=float, default=2e-5)
@click.option('--lr_min', type=int, default=0)
@click.option('--warm_up_ratio', type=float, default=0.05)
@click.option('--device', type=str, default='mps')
def benchmark(context_length, d_model, d_ff,  num_layers, num_heads, warmup_steps, steps, lr_max, lr_min, warm_up_ratio, device):
    Tw=int(warm_up_ratio*steps)
    Tc=steps
    set_seed(42)
    vocab_size=10000
    batch_size=4
    transformer=get_model(context_length, d_model, d_ff, num_layers, num_heads).to(device)
    train_data=get_random_data(100*context_length, vocab_size)
    opt=optimizer.AdamW(transformer.parameters())
    train_steps(warmup_steps, lr_max, lr_min, Tw, Tc, transformer, opt, train_data, batch_size, context_length, device, True)
    elpased=timeit.timeit(
        train_steps(steps, lr_max, lr_min, Tw, Tc, transformer, opt, train_data, batch_size, context_length, device),
        number=3
    )
    console.print(Panel.fit(f"[bold red]Avg Training Time: [/bold red]{elpased}"))

if __name__=='__main__':
    benchmark()