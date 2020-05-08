# -*- coding: utf-8 -*-
import re
import unicodedata

def get_nested_dict_value(d, key):
    """Uses '.'-splittable string as key to access nested dict."""
    try:
        val = d[key]
    except KeyError:
        key = key.replace("->", ".")  # make sure no -> left
        try:
            key_prefix, key_suffix = key.split('.', 1)
        except ValueError:   # not enough values to unpack
            raise KeyError

        val = get_nested_dict_value(d[key_prefix], key_suffix)

    return val


# source https://github.com/django/django/blob/master/django/utils/text.py#L394
def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    # don't just remove special chars, but keep them as spaces -> dashes
    value = re.sub('[^a-zA-Z0-9- _ .\n\.]', ' ', value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower()).strip()
    return re.sub(r'[-\s]+', '-', value)
