import warnings
import numbers
import numpy as np

import functools
# real-time printout
print = functools.partial(print, flush=True)

def print_matrix(keyword, matrix, nwidth=6, nind=0, digits=[13,8,'f'],
                 trans=False, dtype=float):
    r"""
    Print multi dimensional array in formatted way.

    Parameters
        keyword : string of the title line
        matrix : multi dimensional array
        nwidth : int, optional
            number of columns to print in one block (default: 6)
        nind : int for whether to print the row/column index
            larger than 0 to print index (default: 0)
        digits : list of int, optional
            [width, precision, notation], where
            width : total width of each number (default: 13)
            precision : number of digits after decimal point (default: 8)
            notation : 'f' for fixed-point, 'e' for scientific (default: 'f')
        trans : bool, optional
            whether to transpose the last two dimensions (default: False)
        dtype : data type, optional
            data type to convert the input matrix (default: float)
    """

    if '\n' in keyword[-3:]: keyword = keyword[:-2]
    print(keyword)

    if isinstance(matrix, numbers.Real): matrix = np.array([matrix])
    elif isinstance(matrix, list): matrix = np.array(matrix)

    # transpose the last two dimensions
    if trans: matrix = np.einsum('...ij->...ji', matrix)

    width, precision, notation = digits

    ndim = matrix.ndim
    if ndim == 1: # 1d array
        if nwidth==0: nwidth = len(matrix)
        for n in range(len(matrix)):
            if nind > 0: # column index
                #print('%13d ' % n, end='')
                print(f'{n:{width}d} ', end='')
                if (n+1)%nwidth==0: print('')
            #print('%13.8f ' % matrix[n], end='')
            print(f'{matrix[n]:{width}.{precision}{notation}} ', end='')
            if (n+1)%nwidth==0: print('')
        print('\n')

    elif ndim == 2: # 2d array
        nrow, ncol = matrix.shape
        if nwidth==0:
            nloop = 1
        else:
            nloop = ncol//nwidth
            if nloop*nwidth<ncol: nloop += 1

        width2 = len(str(nrow)) + 1

        for n in range(nloop):
            s0, s1 = n*nwidth, (n+1)*nwidth
            if s1>ncol or nwidth==0: s1 = ncol

            if nind > 0: # column index
                for c in range(s0, s1):
                    #print('%13d ' % (c+1), end='')
                    print(f'{(c+1):{width}d} ', end='')
                print('')

            for r in range(nrow):
                if nind > 0: # row index
                    #print('%3d ' % (r+1), end='')
                    print(f'{(r+1):{width2}d} ', end='')
                for c in range(s0, s1):
                    #print('%13.8f ' % matrix[r,c], end='')
                    print(f'{matrix[r,c]:{width}.{precision}{notation}} ', end='')
                print('')

            if nind == 0: # blank line if without column index
                print('')

    elif ndim == 3: # 3d array
        for i in range(matrix.shape[0]):
            print_matrix(keyword+' '+str(i+1)+' ', matrix[i], nwidth, nind, digits)

    elif ndim == 4: # 4d array
        for i in range(matrix.shape[0]):
            print_matrix(keyword+' '+str(i+1)+' ', matrix[i], nwidth, nind, digits)

    elif ndim == 5: # 5d array
        n1, n2, n3 = matrix.shape[:3]
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    print_matrix(keyword+' i: '+str(i+1)+'  j: '+str(j+1)+'  k: '+str(k+1)+'  ', matrix[i, j, k], nwidth, nind, digits)

    elif ndim == 6: # 6d array
        n1, n2, n3, n4 = matrix.shape[:4]
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    for l in range(n4):
                        print_matrix(keyword+' i: '+str(i+1)+'  j: '+str(j+1)+'  k: '+str(k+1)+'  l: '+str(l+1)+'  ', matrix[i, j, k, l], nwidth, nind, digits)

    else:
        warnings.warn('the matrix has higher dimension than this funciton can handle.')


def print_statistics(keyword, array, digits=[4,4]):
    r"""
    Print mean value and standard deviation of a 1D array.

    Parameters
        keyword : string of the title line
        array : 1D array
        digits : list of int, optional
            [precision_mean, precision_std], where
            precision_mean : number of digits after decimal point for mean (default: 4)
            precision_std : number of digits after decimal point for std (default: 4)
    """
    v_mean = np.mean(array)
    v_std = np.std(array) / np.sqrt(len(array))

    if keyword[-1] != ':': keyword += ':'
    print(keyword + f' {v_mean:.{digits[0]}f} ± {v_std:.{digits[1]}f}')


