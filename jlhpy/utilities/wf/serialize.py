#!/usr/bin/env python
#
# bounding_spher.py
#
# Copyright (C) 2020 IMTEK Simulation
# Author: Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from inspect import getmembers, isfunction
import inspect
import logging
import dill

# https://medium.com/@emlynoregan/serialising-all-the-functions-in-python-cd880a63b591
# https://stackoverflow.com/questions/1253528/is-there-an-easy-way-to-pickle-a-python-function-or-otherwise-serialize-its-cod
# https://www.semicolonworld.com/question/59487/how-to-pickle-a-python-function-with-its-dependencies
# https://stackoverflow.com/questions/45616584/serializing-an-object-in-main-with-pickle-or-dill
# https://stackoverflow.com/questions/11866944/how-to-pickle-functions-classes-defined-in-main-python/19428760

# issue with recurse and imports:
# https://github.com/uqfoundation/dill/issues/219


def _fixed_create_function(fcode, fglobals, fname=None, fdefaults=None,
                     fclosure=None, fdict=None, fkwdefaults=None):
    # same as FunctionType, but enable passing __dict__ to new function,
    # __dict__ is the storehouse for attributes added after function creation
    if fdict is None: fdict = dict()
    FunctionType = __import__('types').FunctionType
    func = FunctionType(fcode, fglobals or dict(), fname, fdefaults, fclosure)
    func.__dict__.update(fdict) #XXX: better copy? option to copy?
    if fkwdefaults is not None:
        func.__kwdefaults__ = fkwdefaults

    # THE WORKAROUND:
    # if the function was serialized without recurse, fglobals would actually contain
    # __builtins__, but because of recurse only the referenced modules/objects
    # end up in fglobals and we are missing the important __builtins__
    if "__builtins__" not in func.__globals__:
        func.__globals__["__builtins__"] = globals()["__builtins__"]
    return func


dill._dill._create_function = _fixed_create_function


def get_module_member_list(module):
    """Get list of objects within 'module'."""
    # logger = logging.getLogger(__name__)

    def predicate(another_obj):
        return inspect.getmodule(another_obj) == module

    obj_list = inspect.getmembers(module, predicate)
    return obj_list


def serialize_module_obj(obj):
    """Serialize all objects within module of 'obj' along with 'obj'."""
    logger = logging.getLogger(__name__)
    module = inspect.getmodule(obj)
    logger.info("{:s} belongs to {:s}".format(obj.__name__, module.__name__))

    # list of 'name', 'object' tuples
    member_list = get_module_member_list(module)
    member_dict = dict(member_list)
    logger.info("{} in same module as {:s}".format(
        list(member_dict.keys()), obj.__name__))

    # a hack in order to get dill to serialize other objects within same module
    # as 'obj' as well if 'obj' depends on them.
    # dill looks for dependencies only within __main__
    main_backup = {}
    main_deletes = []

    import __main__

    for name, module_member in member_dict.items():
        if hasattr(__main__, name):
            main_backup[name] = getattr(__main__, name)
        else:
            main_deletes.append(name)

        if hasattr(module_member, '__module__'):
            module_member.__module__ = __main__.__name__
        setattr(__main__, name, module_member)

    func_str = dill.dumps(
        getattr(__main__, obj.__name__),
        protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)

    # reset __module__ attribute:
    for name, module_member in member_dict.items():
        if hasattr(module_member, '__module__'):
            module_member.__module__ = module.__name__

    for name, main_member in main_backup.items():
        setattr(__main__, name, main_member)

    for name in main_deletes:
        delattr(__main__, name)

    return func_str


def serialize_obj(obj):
    """Serialize an object without following any dependencies."""
    func_str = dill.dumps(
        obj, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
    return func_str
