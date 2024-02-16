def consolidate_labels(labels):

    """Consolidates labels by assigning consecutive integers starting from 0,
    handling the presence of the label 0.

    Args:
        labels (array-like): An array of labels.

    Returns:
        list: A list of consolidated labels with the same shape as the input.

    Raises:
        ValueError: If any label is not an integer.

    Examples:
        >>> consolidate_labels([6, 2, 0, 2, 1])
        [3, 2, 0, 2, 1]

        >>> consolidate_labels([3, 2, 1, 0])
        [3, 2, 1, 0]

        >>> consolidate_labels([9, 9, 2, 2, 4])
        [3, 3, 1, 1, 2]
    """

    try:
        if 0 in labels:
            dictionary_labels = {old: new for new, old in enumerate(sorted(set(labels)), start=0)}

        else:
            dictionary_labels = {old: new for new, old in enumerate(sorted(set(labels)), start=1)}

        new_labels = [dictionary_labels[label] for label in labels]

        return new_labels
    except:
        raise TypeError("Labels must be integers.")