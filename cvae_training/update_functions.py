import torch
import logging

from loss_functions import elbo_loss, TripletSemiHardLoss


def get_train_step(model, optimizer, triplets_loss_const=0.0, device="cuda:0"):
    """
    returns a training step for ignite, for training a model for the equation:
    X_t = D(E(X_t, t))
    the following losses are calculated:
    - ELBO (==reconstruction+posterior)
    - triplets
    """
    logger = logging.getLogger(__name__)

    def train_step(engine, batch):
        # we accept engine to fit standard ignite signature
        model.train()
        model.to(device)
        optimizer.zero_grad()

        x, shader_indices, ts = batch
        x = x.to(device)  # BxCxHxW, images generated
        shader_indices = shader_indices.to(device)  # B, indices of shaders used to generate these images
        ts = ts.to(device, dtype=torch.float32)  # B, the t value of each image

        # mu and logvar are the embedding
        recon, mu, logvar = model(x, ts)
        # mu,logvar - tuple of 2 (B, model_embedding_size, h, w) for z_bottom, z_top

        # [-1,1] range to [0,1] range
        recon = (recon * 0.5) + 0.5
        x = (x * 0.5) + 0.5

        # all_metrics_to_record is later logged to wandb
        all_metrics_to_record = dict()
        all_metrics_to_record['loss'] = 0
        all_metrics_to_record['elbo'] = elbo_loss(x, recon, mu, logvar).mean()
        all_metrics_to_record['loss'] += all_metrics_to_record['elbo']
        if triplets_loss_const:
            # calculating triplets loss requires pairwise distance. for efficiency, we AveragePool first
            # then we cat mu/logvar/across the hierarchy.
            # the vector is normalized in the triplets loss so the model could learn to encode time-related information
            # into the magntitude, but that would increase posterior L, and investigating alternatives is out of scope
            global_embedding = torch.cat([mu[0].mean(axis=(2, 3)),
                                          mu[1].mean(axis=(2, 3)),
                                          logvar[0].mean(axis=(2, 3)),
                                          logvar[1].mean(axis=(2, 3)),
                                          ], dim=1)

            all_metrics_to_record['triplets'] = TripletSemiHardLoss(shader_indices, global_embedding, device=device)
            all_metrics_to_record['loss'] += triplets_loss_const*all_metrics_to_record['triplets']
            logging.info(f"triplets loss (no const): {all_metrics_to_record['triplets']}")

        logger.info(f"elbo loss: {all_metrics_to_record['elbo']}")
        logger.info(f"loss: {all_metrics_to_record['loss']}")
        all_metrics_to_record['loss'].backward()
        optimizer.step()
        logger.info("done with step")
        all_metrics_to_record['y'] = x
        all_metrics_to_record['y_pred'] = recon
        return all_metrics_to_record
    return train_step


def get_val_step(model, triplets_loss_const=0.0, device="cuda:0"):
    """
    Essentially the same as the training step but without updating the model.
    For further documentation is get_train_step
    """
    logger = logging.getLogger(__name__)

    def val_step(engine, batch):
        # we accept engine to fit standard ignite signature
        model.eval()
        model.to(device)

        x, shader_indices, ts = batch
        x = x.to(device)  # BxCxHxW, images generated
        shader_indices = shader_indices.to(device)  # B, indices of shaders used to generate these images
        ts = ts.to(device, dtype=torch.float32)  # B, the t value of each image

        # mu and logvar are the embedding
        recon, mu, logvar = model(x, ts)
        # mu,logvar - tuple of 2 (B, model_embedding_size, h, w) for z_bottom, z_top

        # [-1,1] range to [0,1] range
        recon = (recon * 0.5) + 0.5
        x = (x * 0.5) + 0.5

        # all_metrics_to_record is later logged to wandb
        all_metrics_to_record = dict()
        all_metrics_to_record['loss'] = 0
        all_metrics_to_record['elbo'] = elbo_loss(x, recon, mu, logvar).mean()
        all_metrics_to_record['loss'] += all_metrics_to_record['elbo']
        if triplets_loss_const:
            global_embedding = torch.cat([mu[0].mean(axis=(2, 3)),
                                          mu[1].mean(axis=(2, 3)),
                                          logvar[0].mean(axis=(2, 3)),
                                          logvar[1].mean(axis=(2, 3)),
                                          ], dim=1)

            all_metrics_to_record['triplets'] = TripletSemiHardLoss(shader_indices, global_embedding, device=device)
            all_metrics_to_record['loss'] += triplets_loss_const*all_metrics_to_record['triplets']

        logger.info(f"val loss: {all_metrics_to_record['loss']}")
        all_metrics_to_record['y'] = x
        all_metrics_to_record['y_pred'] = recon
        return all_metrics_to_record
    return val_step
