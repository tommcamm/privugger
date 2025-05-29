import pytensor
import pytensor.tensor as at
import numpy as np


@pytensor.compile.ops.as_op(itypes=[at.lvector, at.lvector], otypes=[at.
    lvector])
def method(ratings, flip):

    def netflix_anonymisation(ratings, flip):
        NMOVIES = 30
        for i in range(0, NMOVIES):
            if ratings[i] != 0 and flip[i] != 0:
                ratings[i] = flip[i] - 1
        return np.array(ratings)
    return netflix_anonymisation(ratings, flip)
