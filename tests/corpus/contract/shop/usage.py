from .models import Item


def fp_objects():
    return Item.objects


def fp_meta():
    return Item._meta


def fp_pk_on_instance():
    item = Item()
    return item.pk


def real_bug():
    return Item.totally_made_up_attribute
