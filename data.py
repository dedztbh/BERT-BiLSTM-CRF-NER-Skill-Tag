def tuple_array_to_ndarray(tuple_array):
    return tuple_array_transpose(tuple_array)


def ndarray_to_tuple_array(ndarray):
    return tuple_array_transpose(ndarray)


def tuple_array_transpose(m):
    return list(zip(*m))


if __name__ == '__main__':
    a = [(1, 2), (4, 5), (6, 7)]

    assert a == tuple_array_to_ndarray(ndarray_to_tuple_array(a))
    assert a == ndarray_to_tuple_array(tuple_array_to_ndarray(a))
