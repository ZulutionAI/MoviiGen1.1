import torch

# cuda impl of hungarian method
# from torch_linear_assignment import batch_linear_assignment
# from torch_linear_assignment import assignment_to_indices


def cosine_optimal_transport(X, Y):
    """
    Compute optimal transport between two sets of vectors using cosine distance.

    Parameters:
    X: torch.Tensor of shape (n, d)
    Y: torch.Tensor of shape (m, d)

    Returns:
    P: optimal transport plan matrix of shape (n, m)
    """
    # Normalize input vectors
    X_norm = X / torch.norm(X, dim=1, keepdim=True)
    Y_norm = Y / torch.norm(Y, dim=1, keepdim=True)

    # Compute cost matrix using matrix multiplication (cosine similarity)
    C = -torch.mm(X_norm, Y_norm.t())  # negative because we want to minimize distance

    assignment = batch_linear_assignment(C.unsqueeze(dim=0))
    matching_pairs = assignment_to_indices(assignment)

    return C, matching_pairs