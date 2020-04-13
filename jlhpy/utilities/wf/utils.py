def get_nested_dict_value(d, key):
    """Uses '.'-splittable string as key to access nested dict."""
    try:
        val = d[key]
    except KeyError:
        key = key.replace("->", ".")  # make sure no -> left
        try:
            key_prefix, key_suffix = key.split('.',1)
        except ValueError:   # not enough values to unpack
            raise KeyError

        val = get_nested_dict_value(d[key_prefix], key_suffix)

    return val
