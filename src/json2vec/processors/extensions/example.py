from json2vec.processors.base import register


@register.transformation
def default(item):
    return item
