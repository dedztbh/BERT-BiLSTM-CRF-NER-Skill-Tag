def tuple_array_to_ndarray(tuple_array, len_tuple):
    result = [[] for _ in range(len_tuple)]
    for t in tuple_array:
        for i, item in enumerate(t):
            result[i].append(item)
    return result


def ndarray_to_tuple_array(ndarray, len_tuple=None):
    if len_tuple is None:
        len_tuple = len(ndarray)
    assert len(ndarray) == len_tuple
    result = []
    for c in range(len(ndarray[0])):
        array_to_become_tuple = []
        for r in range(len_tuple):
            array_to_become_tuple.append(ndarray[r][c])
        result.append(tuple(array_to_become_tuple))
    return result
