from myapp.models import User


def list_users():
    qs = User.objects.all()
    return list(qs)


def make_user():
    u = User(username="x", email="x@x")
    return u.bogus_typo
