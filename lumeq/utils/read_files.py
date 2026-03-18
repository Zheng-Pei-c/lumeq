from lumeq import np

def istype(data, dtype=float):
    r"""
    Check if the data can be converted to the given data type.
    """
    try:
        f = dtype(data)
        return True
    except ValueError:
        return False


def read_time(filename):
    r"""
    Read the total wall time from the output file.
    """
    keyword = 'total wall time:'
    with open(filename, 'r') as infile:
        for line in infile:
            if line.find(keyword) >= 0:
                time = float(line.split(keyword)[1].split('sec')[0])
                if float(time) / 3600 > 1: time = str('{:.2f} h'.format(float(time) / 3600))
                elif float(time) / 60 > 1: time = str('{:.2f} min'.format(float(time) / 60))
                return time


def read_number(filename, keyword, n=-1, o=1, dtype=int):
    r"""
    Read numbers from the output file.

    Args:
        filename (str): The output file name
        keyword (str): The keyword to search for the target lines
        n (int): when n>=0: the nth number in the line (0-based)
            when n==-1: o is the begin and end indices of the number in the line
        o (int or tuple): o takes 1 or -1 for the reading order when n>=0
            when n == -1: o takes the begin and end indices
        dtype (type): Output data type, e.g. int, float, or str.

    Returns:
        numbers (numpy.ndarray): Numbers converted to the requested data type.
    """
    numbers = []
    with open(filename, 'r') as infile:
        for line in infile:
            if line.find(keyword) >= 0:
                if n == -1:
                    numbers.append(dtype(line[o[0]:(o[0]+o[1])]))
                else:
                    numbers.append(dtype(line.split()[n*o]))

    if len(numbers) == 1: numbers = numbers[0]
    return np.array(numbers, dtype=dtype)


def read_array(filename, keyword=None, nline=0, ncol=4, nrange=[0,4],
               dtype=float, same=True):
    r"""
    Read an array from the output file.

    Args:
        filename (str): The output file name
        keyword (str): The keyword to search for the target lines
            when keyword is None, read the whole file
        nline (int): The number of lines to read after the keyword line
        ncol (int): The expected number of columns in each line
        nrange (list): Begin and end indices of the target data in each line.
        dtype (type): Output data type, e.g. int, float, or str.
        same (bool): When same is True, only read the lines with all data of the same type

    Returns:
        array (numpy.ndarray): Array converted to the requested data type.
    """
    array = []
    def kernel(line):
        data = line.split()
        if len(data) == ncol:
            issame = True
            if same:
                for i in range(len(data)):
                    try:
                        f = dtype(data[i])
                    except:
                        issame = False
            if (not same) or issame:
                for i in range(nrange[0], nrange[1]):
                    try:
                        array.append(dtype(data[i]))
                    except: # encounter a string
                        pass

    with open(filename, 'r') as infile:
        if keyword is None:
            for line in infile:
                kernel(line)
        else:
            for line in infile:
                if line.find(keyword) >= 0:
                    kernel(line)
                    for i in range(nline):
                        kernel(next(infile))

    return np.array(array)


def read_matrix(filename, nrow, ncol, keyword, nwidth=6, nind=0, nskip=0,
                dtype=float):
    r"""
    Read a matrix from the output file.

    Args:
        filename (str): The output file name
        nrow (int): The number of rows of the matrix
        ncol (int): The number of columns of the matrix
        keyword (str): The keyword to search for the target lines
        nwidth (int): The number of columns in each batch
            when nwidth == -1, nwidth = ncol
        nind (int): The number of index columns at the beginning of each row
        nskip (int): The number of rows to skip after the keyword line
        dtype (type): Output data type, e.g. int, float, or str.

    Returns:
        matrix (numpy.ndarray): Matrix with shape ``(nrow, ncol)``.
    """
    if nwidth == -1: nwidth = ncol

    nbatch = ncol // nwidth
    if nbatch * nwidth < ncol: nbatch += 1

    if nrow > 1:
        matrices = []

        with open(filename, 'r') as infile:
            for line in infile:
                if line.find(keyword) >= 0:

                    for n in range(nskip): # skip rows
                        line = next(infile)

                    matrix = np.zeros((nrow, ncol), dtype=dtype)
                    for k in range(nbatch):
                        if k > 0: line = next(infile)
                        elif nind > 0: line = next(infile) # skip the top index
                        for i in range(nrow):
                            data = next(infile).split()
                            for d in range(len(data)-nind):
                                if k*nwidth+d < ncol:
                                    matrix[i, k*nwidth+d] = dtype(data[d+nind])
                    matrices.append(matrix)

        if len(matrices) == 1: matrix = matrices[0]
        else: matrix = np.array(matrices)

        return matrix
    elif nrow == 1:
        matrices = []

        with open(filename, 'r') as infile:
            for line in infile:
                if line.find(keyword) >= 0:

                    matrix = np.zeros(ncol, dtype=dtype)
                    for k in range(nbatch):
                        if nind > 0: line = next(infile) # skip the top index
                        data = next(infile).split()
                        for d in range(len(data)-nind):
                            if k*nwidth+d < ncol:
                                matrix[k*nwidth+d] = dtype(data[d+nind])

                        for n in range(nskip): # skip rows
                            line = next(infile)
                    matrices.append(matrix)

        if len(matrices) == 1: matrix = matrices[0]
        else: matrix = np.array(matrices)

        return matrix
