from json2vec.processors.base import register


@register
def default(item):
    return item
