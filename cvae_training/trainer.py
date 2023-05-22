from pathlib import Path
import numpy as np
import torch
import os
from glob import glob

import wandb
import ignite
from ignite.engine import Events
from ignite.metrics import Average, FID
from ignite.handlers import Checkpoint, DiskSaver

from wandb_utils import remove_old_wandb_media
from update_functions import get_train_step, get_val_step
from image_generation.shaders.dataloading import get_dataloader, get_dataloader_test
from model import Model
from cvae_training.wandb_utils import tensor2np, path_to_wandb_dir


class Trainer:
    """
    this class manages the entire training process
    """
    def __init__(self, args):
        wandb.init(
            project="shaders",
            config={
                "lr": args.lr,
                "architecture": f"hierarchical {'C' if not args.ignore_time else ''}VAE",
                "latent_dim": args.latent_dim,
                "dataset": "twigl" if args.dataset else "shadertoy",
                "resolution": args.resolution,
                "virtual_dataset_size": args.epoch_length,
                "limit_shaders": args.limit_shaders,
                "epochs": args.epochs,
                "seed": args.seed
            },
            resume="must" if args.resume_from else None,
            id=args.resume_from if args.resume_from else None
        )

        # data
        self.device = torch.device(args.device)
        self.num_epochs = args.epochs
        self.train_loader = get_dataloader(twigl=args.dataset, is_train=True, n=args.epoch_length, gpu=0,
                                           batch_size=args.batch_size, limit_shaders=args.limit_shaders,
                                           resolution=args.resolution)
        self.val_loader = get_dataloader(twigl=args.dataset, is_train=False, n=args.epoch_length, gpu=0,
                                         batch_size=args.batch_size, limit_shaders=args.limit_shaders,
                                         resolution=args.resolution)

        # weights
        self.model = Model(latent_dim=args.latent_dim, device=self.device, use_time=not args.ignore_time)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # core update loop
        train_step = get_train_step(self.model, self.optimizer, triplets_loss_const=args.triplets, device=self.device)
        val_step = get_val_step(self.model, triplets_loss_const=args.triplets, device=self.device)
        self.trainer = ignite.engine.Engine(train_step)
        self.validator = ignite.engine.Engine(val_step)

        # metric tracking - train
        Average(output_transform=lambda d: d["loss"]).attach(self.trainer, "loss")
        Average(output_transform=lambda d: d["elbo"]).attach(self.trainer, "elbo")
        Average(output_transform=lambda d: d["triplets"] if "triplets" in d else 0).attach(self.trainer, "triplets")
        FID(device=self.device).attach(self.trainer, "fid")

        # metric tracking - val
        Average(output_transform=lambda d: d["loss"]).attach(self.validator, "val loss")
        Average(output_transform=lambda d: d["elbo"]).attach(self.validator, "val elbo")
        Average(output_transform=lambda d: d["triplets"] if "triplets" in d else 0).attach(self.validator, "val triplets")
        FID(device=self.device).attach(self.validator, "val fid")

        if args.resume_from:
            self.restore_from_checkpoint(args.full_run_dir)

        # add a lot of callbacks for logging metrics and saving model
        @self.trainer.on(Events.ITERATION_STARTED(once=1))
        def first_iteration_started(trainer):
            print(f"{self.trainer.state.iteration} first iteration started")

        @self.trainer.on(Events.ITERATION_COMPLETED(before=10) | Events.ITERATION_COMPLETED(every=2, after=10))
        def iteration_completed(trainer):
            print(f"{self.trainer.state.iteration} iterations completed")

        @self.validator.on(Events.ITERATION_COMPLETED(before=10) | Events.ITERATION_COMPLETED(every=2, after=10))
        def iteration_completed_val(validator):
            print(f"{self.validator.state.iteration} iterations completed val")

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def clear_memory(trainer):
            self.model.clear_grad_memory()

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def print_training_results(trainer):
            metrics = trainer.state.metrics
            print(f"Training Results - Epoch[{trainer.state.epoch}],  Results: {metrics}")

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def print_val_results(trainer):
            self.validator.run(self.val_loader)
            metrics = self.validator.state.metrics
            print(f"Validation Results - Epoch[{trainer.state.epoch}], Results: {metrics}")

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_metrics_wandb(trainer):
            train_metrics = self.trainer.state.metrics
            val_metrics = self.validator.state.metrics

            examples_dict_train = self.infer_example_batch_wandb(loader=self.train_loader)
            examples_dict_val = self.infer_example_batch_wandb(loader=self.val_loader)
            examples_dict_val = {"val_" + k: v for k, v in examples_dict_val.items()}

            example_dict_vid = self.generate_videos_wandb(2, num_variations_each=2)

            # get rid of old media which takes space and was already uploaded. this prevents drive mount errors
            # media has an entire epoch to be synced before it gets deleted
            remove_old_wandb_media()

            wandb.log({"epoch": self.trainer.state.epoch,
                       **train_metrics, **val_metrics,
                       **examples_dict_train, **examples_dict_val,
                       **example_dict_vid})

        # we only add checkpointing at the end because it uses metrics
        self.add_checkpoint_saving_callback()

        @torch.no_grad()
        def test(n_sampled_images=8):
            # prepare data, testing ignite.Engine, and metrics
            test_loader = get_dataloader_test(twigl=args.dataset, gpu=0, batch_size=args.batch_size,
                                              limit_shaders=args.limit_shaders, resolution=args.resolution)

            self.tester = ignite.engine.Engine(val_step)
            Average(output_transform=lambda d: d["loss"]).attach(self.tester, "test loss")
            Average(output_transform=lambda d: d["elbo"]).attach(self.tester, "test elbo")
            Average(output_transform=lambda d: d["triplets"] if "triplets" in d else 0).attach(self.tester,
                                                                                               "test triplets")
            FID(device=self.device).attach(self.tester, "test fid")

            # actually run the test
            self.tester.run(test_loader)
            metrics = self.tester.state.metrics

            # test reconstruction examples
            test_images = self.infer_example_batch_wandb(loader=test_loader)
            test_images = {"test_" + k: v for k, v in test_images.items()}

            # test videos from single non-shader image
            try:
                testing_images_to_videofy = [f"example_image{i}.jpg" for i in range(1, 3)]
                if not os.path.isfile(testing_images_to_videofy[0]):
                    testing_images_to_videofy = [os.path.join("..", path)
                                                 for path in testing_images_to_videofy]
                test_vids = self.generate_videos_wandb(testing_images_to_videofy, num_variations_each=2)
            except Exception as e:
                print("failed to make vids", e)
                test_vids = dict()

            # test sampled images (unconditional on input)
            try:
                sampled_images = self.model.sample(t=torch.zeros(n_sampled_images, dtype=torch.float32, device=self.device),
                                                   hw=args.resolution,
                                                   sample_size=n_sampled_images)
                unconditional_sampling = [wandb.Image(im, caption=f"test sampled image {i}")
                                          for i, im in enumerate(sampled_images)]
                unconditional_sampling = dict(unconditional_sampling=unconditional_sampling)
            except Exception as e:
                print("failed to make unconditional samples", e)
                unconditional_sampling = dict()

            wandb.log({**metrics, **test_images, **test_vids, **unconditional_sampling})

        # this is a hacky solution to both have the args Trainer got, and have the function be available externally
        self.test = test

    @torch.no_grad()
    def infer_example_batch_wandb(self, loader, k=8):
        """
        loader - dataloader to take samples from
        k - how many samples to show, up to batch_size
        """
        if k > self.train_loader.batch_size:
            print(f"infer_example_batch_wandb: k is too large. only showing bs={self.train_loader.batch_size}")

        # loading input data
        batch = next(iter(loader))
        x, shader_indices, ts = batch
        x = x[:k]
        shader_indices = shader_indices[:k]
        ts = ts[:k]
        x = x.to(self.device)
        ts = ts.to(self.device, dtype=torch.float32)
        shader_names = [self.train_loader.dataset.programs[ind].name for ind in shader_indices]

        # infer
        self.model.eval()
        recon, _, _ = self.model(x, ts)
        x = tensor2np(x)
        recon = tensor2np(recon)

        # changing results to wandb format
        shader_indices = shader_indices.cpu().numpy()
        ts = ts.detach().cpu().numpy().round(4)
        examples_gt = [wandb.Image(xx, caption=f"shader {ind} ({name}) at t={t}") for xx, ind, t, name in
                       zip(x, shader_indices, ts, shader_names)]
        examples_recon = [wandb.Image(rr, caption=f"shader {ind} ({name}) at t={t}") for rr, ind, t, name in
                          zip(recon, shader_indices, ts, shader_names)]
        examples_dict = dict(examples_gt=examples_gt, examples_recon=examples_recon)
        return examples_dict

    @torch.no_grad()
    def generate_videos_wandb(self, images, num_variations_each=1):
        """
        images - either int, up to batch_size, number of random val shader images to select,
                 or; list of paths, to images that should be video-fied
        num_variations_each - how many videos to make of each image
        """
        # getting images and names
        if type(images) is int:
            x, shader_indices, _ = next(iter(self.val_loader))
            x = x[:images]
            names = [self.val_loader.dataset.programs[ind].name for ind in shader_indices[:images]]
        else:
            import cv2
            x = [torch.tensor(cv2.resize(cv2.imread(path), (256, 256)).astype(np.float32)/255).permute(2, 0, 1)
                 for path in images]
            names = [Path(path).stem for path in images]

        # generating videos
        all_videos = {}
        for i, (image, name) in enumerate(zip(x, names)):
            videos = self.model.predict_video_from_image(image, num_variations_to_generate=num_variations_each)

            all_videos.update({"video": wandb.Video(vid, fps=25, caption=f"video_{name}_variation{j}")
                               for j, vid in enumerate(videos)})
        return all_videos

    def add_checkpoint_saving_callback(self):
        """
        checkpoints both the model and everything needed to continue training
        """
        to_save = {
            "train_engine": self.trainer,
            "val_engine": self.validator,
            "model": self.model,
            "optimizer": self.optimizer
        }

        neg_val_loss = lambda engine: -self.validator.state.metrics['val loss']  # Checkpoint saves highest, not lowest
        handler = Checkpoint(to_save, DiskSaver(wandb.run.dir, create_dir=False), score_function=neg_val_loss,
                             score_name="val_loss", n_saved=1)
        self.validator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    def restore_from_checkpoint(self, wandb_dir):
        print(f"attempting to resume run from {wandb_dir}")
        to_load = {
            "train_engine": self.trainer,
            "val_engine": self.validator,
            "model": self.model,
            "optimizer": self.optimizer
        }
        checkpoint_files = glob(os.path.join(wandb_dir, "files", "*.pt"))
        assert len(checkpoint_files) == 1, f"could not find which checkpoint to continue {checkpoint_files}"
        checkpoint = checkpoint_files[0]
        Checkpoint.load_objects(to_load, checkpoint=checkpoint)
        print(f"resuming run from {wandb_dir}")

    def teardown(self):
        """
        clean exit
        """
        print("tearing down")
        wandb.finish()
        self.train_loader.dataset.deconstruct()
        self.val_loader.dataset.deconstruct()

    def run(self):
        try:
            self.trainer.run(data=self.train_loader, max_epochs=self.num_epochs)
            print("training complete successfully!")
        except:
            self.teardown()
            raise

        # testing
        print("starting testing")
        self.test()
        self.teardown()
