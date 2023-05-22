import logging
import torch
import torch.nn.functional as F


def elbo_loss(x, recon, mu, logvar):
    """
    Balanced ELBO loss

    there's a problem with optimizing the ELBO directly. there are 2 components: log likelihood and latent KL divergence
    data negative log likelihood under the model increases linearly with data size (i.e. 256*256*3)
    latent KL divergence increases linearly with latent size
    since 256*256*3 is very large this element dominates the loss.
    to solve this we use mean instead of sum in both cases.
    this means the loss isn't quite ELBO technically, but more balanced and doesn't blow up the loss

    :param x: tensor (B, C, H, W)
    :param recon: tensor (B, C, H, W)
    :param mu: tuple of 2 tensors (B, latent_dim, h, w) where the first element is mean for z_bottom and the second z_top
    :param logvar: tuple of 2 tensors (B, latent_dim, h, w) where the first element is logvar for z_bottom and the second z_top
    :return: tensor (B,), of sample loss
    """
    reconstruction_loss = F.binary_cross_entropy(recon, x, reduction='none').mean(axis=(1, 2, 3))  # changed to mean

    posterior_loss_fn = lambda m, l: -l + 0.5*(torch.exp(l)**2 + m**2 - 1)

    posterior_loss = (posterior_loss_fn(mu[0], logvar[0]).mean(axis=(1, 2, 3)) +
                      posterior_loss_fn(mu[1], logvar[1]).mean(axis=(1, 2, 3))
                      ) / 2.0

    logger = logging.getLogger(__name__)
    logger.info(f"recong loss: {reconstruction_loss[:2]}.., (mean={reconstruction_loss.mean()})")
    logger.info(f"posterior loss: {posterior_loss[:2]}.., (mean={posterior_loss.mean()})")
    return reconstruction_loss + posterior_loss


# the following two functions are taken with minor modifications from
# https://github.com/alfonmedela/triplet-loss-pytorch/blob/b3da5393e41ba0e81aa7a770e279685f8dee57d1/loss_functions/triplet_loss.py
def pairwise_distance_torch(embeddings, device):
    """
    Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances


def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    """
    Computes the triplet loss_functions with semi-hard negative mining.
    The loss_functions encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of embedding vectors.
    Args:
      margin: Float, margin term in the loss_functions definition. Default value is 1.0.

    The only addition from https://github.com/alfonmedela/... is normalizing the embedding
   """

    labels, unnormed_embeddings = y_true, y_pred

    # normalize
    embeddings = unnormed_embeddings / torch.functional.norm(unnormed_embeddings, dim=1, keepdim=True)

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    if num_positives == 0:
        return num_positives

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss
