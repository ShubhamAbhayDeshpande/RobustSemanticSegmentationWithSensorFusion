"""
This file will implement the L1 fusion strategy mentioned in the paper.

Import this file into testing module to use. 

"""

import torch

def L1_norm(source_en_a, source_en_b):
    result = []

    # Calculate the L1 norm of the vectors
    l1_a = torch.sum(torch.abs(source_en_a), dim=3).sum(0)
    l1_b = torch.sum(torch.abs(source_en_b), dim=3).sum(0)

    # Calculate the map for the source image
    mask_value = l1_a + l1_b
    mask_sign_a = l1_a / mask_value
    mask_sign_b = l1_b / mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b

    dimension = source_en_a.shape  # Here the assumption is that, the shapes of the matrix a and b are the same.
    assert source_en_a.shape == source_en_b.shape

    for i in range(dimension[3]):
        temp_matrix = array_MASK_a * source_en_a[0, :, :, i] + array_MASK_b * source_en_b[0, :, :, i]
        result.append(temp_matrix)

    result = torch.stack(result, dim=1)

    return result.view(dimension)