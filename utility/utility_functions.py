import numpy as np
def best_int_split(ratio, total):
    """
    Divides a total into integer shares that best reflects ratio
    # Taken from https://github.com/google/brain-tokyo-workshop/blob/6d0a262171cca7e2e08f901981880e5247b4d677/WANNRelease/prettyNeatWann/utils/utils.py#L57
    Args:
        ratio      - [1 X N ] - Percentage in each pile
        total      - [int   ] - Integer total to split

    Returns:
        intSplit   - [1 x N ] - Number in each pile
    """
    # Handle poorly defined ratio
    if sum(ratio) is not 1:
        ratio = np.asarray(ratio) / sum(ratio)

    # Get share in real and integer values
    floatSplit = np.multiply(ratio, total)
    intSplit = np.floor(floatSplit)
    remainder = int(total - sum(intSplit))

    # Rank piles by most cheated by rounding
    deserving = np.argsort(-(floatSplit - intSplit), axis=0)

    # Distribute remained to most deserving
    intSplit[deserving[:remainder]] = intSplit[deserving[:remainder]] + 1
    return np.array(intSplit, dtype=np.int)