import argparse
from glob import glob
import os

from wandb_utils import remove_old_wandb_runs
from trainer import Trainer
from ignite.utils import manual_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset related
    parser.add_argument('--twigl', help='use twigl dataset instead of shadertoy.', dest='dataset', action='store_true')
    parser.add_argument('--shadertoy', help='use shadertoy dataset instead of twigl.', dest='dataset', action='store_false')
    parser.set_defaults(dataset=False)
    parser.add_argument('--resolution', help='resolution of generated images.', type=int, default=256)
    parser.add_argument('--limit_shaders', help='only use this many of the shaders. if 0, use all', type=int, default=0)

    # model related
    parser.add_argument('--latent_dim', help='.', type=int, default=512)
    parser.add_argument('--ignore_time', help='ignore the time conditioning (so the model is a VAE)',
                        action='store_true')

    # training related
    parser.add_argument('--epochs', help='maximum number of iterations.', type=int, default=50)
    parser.add_argument('--epoch_length', help='every epoch will have this many samples', type=int, default=200)
    parser.add_argument('--batch_size', help='number of images in a mini-batch.', type=int, default=64)
    parser.add_argument('--resume_from', help='resume a previous run from this wandb dir', type=str, default=None)
    parser.add_argument('--lr', help='initial learning rate.', type=float, default=1e-3)
    parser.add_argument('--triplets', help='triplets loss constant. use 0 to disable', type=float, default=0)
    parser.add_argument('--device', help='gpu id/cpu to use.', type=str, default="cpu")
    parser.add_argument('--seed', help='seed (except for shader selection)', type=int, default=0)

    parser.add_argument('--del_prev_runs', help='delete the wandb directory of all previous runs', action='store_true')
    args = parser.parse_args()

    manual_seed(args.seed)

    if args.del_prev_runs:
        remove_old_wandb_runs()

    if args.resume_from:
        runs = glob(os.path.join("wandb", "run-*"))
        run = [run for run in runs if run.endswith("-" + args.resume_from)]
        assert len(run) == 1, f"could not figure out which run to continue from. {runs}"
        args.full_run_dir = run[0]

    trainer = Trainer(args)
    trainer.run()
