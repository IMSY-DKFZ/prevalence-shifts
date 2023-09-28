import torch


class ImpossibleScalingError(Exception):
    """Triggered if scaling is not possible according to requirements."""


def imbalance_ratio(classes: torch.Tensor) -> float:
    """Calculates the imbalance ratio between classes."""
    class_prevalences = torch.bincount(classes)
    min_cases = torch.min(class_prevalences)
    max_cases = torch.max(class_prevalences)
    return max_cases / min_cases


def scale_prevalences_ir(logits: torch.Tensor, classes: torch.Tensor, ir: float = 1., base_seed: int = 42):
    """
    Scales prevalences linearly based on target imbalance ratio, returns a subsampled set of logits and classes
    accordingly.
    """
    # check if given imbalance ratio is valid
    if ir < 1:
        raise ImpossibleScalingError(f'Imbalance ratio below 1 requested ({ir}')
    # set random seed for reproducibility
    torch.manual_seed(seed=int(base_seed + 100 * ir))
    # compute number of samples in each class
    class_prevalences = torch.bincount(classes)
    # compute the original imbalance ratio
    orig_ir = imbalance_ratio(classes)
    # initialize the list of indices for the final subsampled set
    final_indices = []
    # find the minimal and maximal number of samples in each class
    min_cases = torch.min(class_prevalences)
    max_cases = torch.max(class_prevalences)
    # find index of class with maximal number of indices
    max_class = torch.argmax(class_prevalences)
    # iterate over classes
    for i, value in enumerate(class_prevalences):
        if ir >= orig_ir:
            # downsample all but the max_class
            if i != max_class:
                class_prevalences[i] = (class_prevalences[i] * max_cases) / (
                            min_cases * ir)  # undersample smaller classes
                # raise error if class is too small
                if class_prevalences[i] < 9:
                    raise ImpossibleScalingError(f'{torch.bincount(classes)} and ir: {ir}')
        else:
            # calculate the temperature
            temp = (ir - 1) / (orig_ir - 1)
            class_prevalences[i] = min_cases + temp * (class_prevalences[i] - min_cases)
        # get indices of the class
        class_indices = (classes == i).nonzero().squeeze()
        # randomy select a subset of indices of size given by the class prevalence
        new_indices = class_indices[torch.randperm(len(class_indices))[:class_prevalences[i]]]
        # add the subset to the final list
        final_indices.extend(new_indices)
    # stack the final indices
    final_indices = torch.stack(final_indices)
    # return the logits and classes for the final idices
    return logits[final_indices], classes[final_indices]
