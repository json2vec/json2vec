from json2vec.processors.base import shim


@shim(yields=False)
def default(item):
    return item
