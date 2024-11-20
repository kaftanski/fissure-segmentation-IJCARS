import os

import SimpleITK as sitk
import imageio
import numpy as np
import pyamg
import torch
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from torch.nn import functional as F

from utils.visualization import visualize_with_overlay


def compute_laplace_matrix(im: torch.Tensor, edge_weights: str, graph_mask: torch.Tensor = None) -> torch.sparse.Tensor:
    """ Computes Laplacian matrix for an n-dimensional image with intensity weights.

    :param im: input image
    :param edge_weights: either 'binary' for binary probabilities, 'intensity' for intensity difference weights
    :param graph_mask: tensor mask where each zero pixel should be excluded from graph (e.g.: lung-mask)
    :return: Laplacian matrix
    """
    # parameters and image dimensions
    sigma = 8
    lambda_ = 1
    n_nodes = im.numel()

    # create 1D index vector
    ind = torch.arange(n_nodes).view(*im.size())

    # define the graph, one dimension at a time
    A_per_dim = []
    for dim in range(len(im.shape)):
        slices = [slice(None)] * dim

        # define one step forward (neighbors) in the current dimension via the continuous index
        i_from = ind[slices + [slice(None, -1)]].reshape(-1, 1)
        i_to = ind[slices + [slice(1, None)]].reshape(-1, 1)
        ii = torch.cat((i_from, i_to), dim=1)

        if graph_mask is not None:
            # remove edges containing pixels not in the graph-mask
            graph_indices = ind[graph_mask != 0].view(-1)
            # ii = torch.take_along_dim(ii, indices=graph_indices, dim=0)
            ii = torch.stack([edge for edge in ii if edge[0] in graph_indices and edge[0] in graph_indices], dim=0)  # this could be more performant

        if edge_weights == 'intensity':
            # compute exponential edge weights from image intensities
            val = torch.exp(-(torch.take(im, ii[:, 0]) - torch.take(im, ii[:, 1])).pow(2) / (2 * sigma ** 2))
        elif edge_weights == 'binary':
            # 1 if values are the same, 0 if not
            val = torch.where((torch.take(im, ii[:, 0]) == torch.take(im, ii[:, 1])), 1., 0.01)
            # val = (torch.take(im, ii[:, 0]) == torch.take(im, ii[:, 1])).float()
        else:
            raise ValueError(f'No edge weights named "{edge_weights}" known.')

        # create first part of neighbourhood matrix (similar to setFromTriplets in Eigen)
        A = torch.sparse.FloatTensor(ii.t(), val, torch.Size([n_nodes, n_nodes]))

        # make graph symmetric (add backward edges)
        A = A + A.t()
        A_per_dim.append(A)

    # combine all dimensions into one graph
    A = A_per_dim[0]
    for a in A_per_dim[1:]:
        A += a

    # compute degree matrix (diagonal sum)
    D = torch.sparse.sum(A, 0).to_dense()

    # put D and A together
    L = torch.sparse.FloatTensor(torch.cat((ind.view(1, -1), ind.view(1, -1)), 0), .00001 + lambda_ * D,
                                 torch.Size([n_nodes, n_nodes]))
    L += (A * (-lambda_))

    return L


def random_walk(L: torch.sparse.Tensor, labels: torch.Tensor, graph_mask: torch.Tensor = None) -> torch.Tensor:
    """

    :param L: graph laplacian matrix, as created by compute_laplace_matrix
    :param labels: seed points/scribbles for each object. should contain values of [0, 1, ..., N_objects].
        Note: a value of 0 is considered as "unseeded" and not as background. If you want to segment background as well,
        please assign it the label 1.
    :param graph_mask: binary tensor of the same size as labels. will remove any False elements from the optimization,
        these will have a 0 probability for all labels as a result
    :return: probabilities for each label at each voxel. shape: [labels.shape[...], N_objects]
    """
    # linear index tensor
    ind = torch.arange(labels.numel())

    # extract seeded (x_s) and unseeded indices (x_u) and limit to within the mask, if provided
    seeded = labels.view(-1) != 0
    if graph_mask is None:
        graph_mask = torch.tensor(True)

    seeded = torch.logical_and(seeded, graph_mask.view(-1))  # remove seeded nodes outside the mask
    x_s = ind[seeded]
    x_u = ind[torch.logical_and(torch.logical_not(seeded), graph_mask.view(-1))]

    # get blocks from L: L_u (edges between unseeded nodes) and B^T (edges between an unseeded and an seeded node)
    L_u = sparse_cols(sparse_rows(L, x_u), x_u)
    B_T = sparse_rows(sparse_cols(L, x_s), x_u)

    # create seeded probabilities u_s
    u_s = F.one_hot(labels.view(-1)[seeded] - 1).float()

    # solve sparse LSE
    u_u = sparseMultiGrid(L_u, -1 * torch.sparse.mm(B_T, u_s), 1)

    probabilities = torch.zeros(labels.numel(), u_u.shape[-1])
    probabilities[x_s] = u_s
    probabilities[x_u] = u_u
    return probabilities.view(*labels.shape, -1)


def sparseMultiGrid(A, b, iterations):  # A sparse torch matrix, b dense vector
    """ provided function that calls the sparse LSE solver using multi-grid """
    A_ind = A._indices().cpu().data.numpy()
    A_val = A._values().cpu().data.numpy()
    n1, n2 = A.size()
    SC = csr_matrix((A_val, (A_ind[0, :], A_ind[1, :])), shape=(n1, n2))
    ml = pyamg.ruge_stuben_solver(SC, max_levels=6)  # construct the multigrid hierarchy
    # print(ml)                                           # print hierarchy information
    b_ = b.cpu().data.numpy()
    x = b_ * 0
    for i in range(x.shape[1]):
        x[:, i] = ml.solve(b_[:, i], tol=1e-3)
    return torch.from_numpy(x)  # .view(-1,1)


def sparse_rows(S, slice):
    """ provided functions that removes/selects rows from sparse matrices """
    # sparse slicing
    S_ind = S._indices()
    S_val = S._values()
    # create auxiliary index vector
    slice_ind = -torch.ones(S.size(0)).long()
    slice_ind[slice] = torch.arange(slice.size(0))
    # def sparse_rows(matrix,indices):
    # redefine row indices of new sparse matrix
    inv_ind = slice_ind[S_ind[0, :]]
    mask = (inv_ind > -1)
    N_ind = torch.stack((inv_ind[mask], S_ind[1, mask]), 0)
    N_val = S_val[mask]
    S = torch.sparse.FloatTensor(N_ind, N_val, (slice.size(0), S.size(1)))
    return S


def sparse_cols(S, slice):
    """ provided functions that removes/selects cols from sparse matrices """
    # sparse slicing
    S_ind = S._indices()
    S_val = S._values()
    # create auxiliary index vector
    slice_ind = -torch.ones(S.size(1)).long()
    slice_ind[slice] = torch.arange(slice.size(0))
    # def sparse_rows(matrix,indices):
    # redefine row indices of new sparse matrix
    inv_ind = slice_ind[S_ind[1, :]]
    mask = (inv_ind > -1)
    N_ind = torch.stack((S_ind[0, mask], inv_ind[mask]), 0)
    N_val = S_val[mask]
    S = torch.sparse.FloatTensor(N_ind, N_val, (S.size(0), slice.size(0)))
    return S
