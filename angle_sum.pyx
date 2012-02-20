#
#   angle_sum - algorithm to sum an array along arbitrary angles
#   Copyright (C) 2012 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
cimport cython

dtype = np.int
ctypedef np.int_t dtype_t 
outdtype = np.int
ctypedef np.int_t outdtype_t 

@cython.boundscheck(False)
def angle_sum(np.ndarray[dtype_t, ndim=2] m not None,
        float angle, float aspect=1., float binsize=0.):
    """Compute the sum of a 2D array along an rotated axis.

    Parameters
    ----------
    m : array_like, shape(N, M)
        2D input array to be summed
    angle : float
        The angle of the summation direction defined such that:
            angle_sum(m, angle=0) == np.sum(m, axis=0)
            angle_sum(m, angle=np.pi/2) == np.sum(m, axis=1)
    aspect : float
        The input bin aspect ratio (second dimension/first dimension).
    binsize : float
        The output bin size in units of the first input dimension bin
        size. If no binsize is given, it defaults to the "natural bin
        size" which is the larger projection of the two input bin sizes
        onto the output dimension (the axis perpendicular to the
        summation axis).

    Returns
    -------
    out : ndarray, shape(K)
        The sum of `m` along the axis at `angle`.

    Notes
    -----
    The summation angle is relative to the first dimension.

    For 0<=angle<=pi/2 the value at [0,0] ends up in the first bin and
    the value at [-1,-1] ends up in the last bin.

    For angle=3/4*pi the summation is along the diagonal.
    For angle=3/4*pi the summation is along the antidiagonal.
   
    The origin of the rotation is the [0,0] index. This determines the
    bin rounding.

    Up to index flipping, limits, rounding, offset and the definition of
    `angle` the output `o` is:

    .. math::
      o_k = \\sum_l m_{i,j/a}
      i(l,k) = \\cos(\\alpha) l - \\sin(\\alpha) k
      j(l,k) = \\sin(\\alpha) l + \\cos(\\alpha) k

    There is no interpolation and artefacts are likely.

    The full array sum is strictly conserved.

    Examples
    --------
    >>> m = np.arange(9.).reshape((3, 3))
    >>> np.all(angle_sum(m, 0) == m.sum(axis=0))
    True
    >>> np.all(angle_sum(m, np.pi/2) == m.sum(axis=1))
    True
    >>> np.all(angle_sum(m, np.pi) == m.sum(axis=0)[::-1])
    True
    >>> np.all(angle_sum(m, 3*np.pi/2) == m.sum(axis=1)[::-1])
    True
    >>> np.all(angle_sum(m, 2*np.pi) == m.sum(axis=0))
    True
    >>> np.all(angle_sum(m, -np.pi/2) == 
    ...        angle_sum(m, 3*np.pi/2))
    True
    >>> d1 = np.array([0, 4, 12, 12, 8]) # antidiagonal
    >>> d2 = np.array([2, 6, 12, 10, 6]) # diagonal
    >>> np.all(angle_sum(m, np.pi/4) == d1)
    True
    >>> np.all(angle_sum(m, 3*np.pi/4) == d2)
    True
    >>> np.all(angle_sum(m, 5*np.pi/4) == d1[::-1])
    True
    >>> np.all(angle_sum(m, 7*np.pi/4) == d2[::-1])
    True
    >>> np.all(angle_sum(m, 0, aspect=2, binsize=1) == 
    ...        np.array([9, 0, 12, 0, 15]))
    True
    >>> np.all(angle_sum(m, 0, aspect=.5, binsize=1) == 
    ...        np.array([9, 12+15]))
    True
    >>> np.all(angle_sum(m, 0, aspect=.5) == m.sum(axis=0))
    True
    >>> np.all(angle_sum(m, np.pi/2, aspect=2, binsize=1) ==
    ...        m.sum(axis=1))
    True
    >>> m2 = np.arange(1e6).reshape((100, 10000))
    >>> np.all(angle_sum(m2, 0) == m2.sum(axis=0))
    True
    >>> np.all(angle_sum(m2, np.pi/2) == m2.sum(axis=1))
    True
    >>> angle_sum(m2, np.pi/4).shape
    (10099,)
    >>> angle_sum(m2, np.pi/4).sum() == m2.sum()
    True
    """
    cdef np.ndarray[np.int_t, ndim=2] i, j
    cdef np.ndarray[np.float64_t, ndim=2] k
    cdef np.ndarray[outdtype_t, ndim=1] out
    cdef int km, kp
    cdef unsigned int im = m.shape[0]
    cdef unsigned int jm = m.shape[1]

    if binsize == 0:
        binsize = max(abs(np.cos(angle)*aspect),
                      abs(np.sin(angle)))
    # first axis needs to be inverted for the angle convention
    # to make work
    m = m[::-1]
    i, j = np.ogrid[:m.shape[0], :m.shape[1]]
    # output coordinate
    k = np.cos(angle)*j*aspect-np.sin(angle)*i
    # output bin index
    k = np.floor(k/binsize+.5)
    # output array size
    cx, cy = (0, 0, -1, -1), (0, -1, 0, -1)
    km = int(k[cx, cy].min())
    kp = int(k[cx, cy].max())
    #assert k.min() == km
    #assert k.max() == kp
    out = np.zeros([kp-km+1,], dtype=outdtype)
    for ii in range(im):
        for jj in range(jm):
            out[<unsigned int>k[ii, jj]-km] += m[ii, jj]
    return out
