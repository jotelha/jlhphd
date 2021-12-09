import random
import string

def random_alphanumeric_string(length=4):
    return ''.join(random.sample((string.ascii_uppercase+string.digits),length))