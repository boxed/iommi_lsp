from django.db import models


class Timestamped(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Book(Timestamped):
    title = models.CharField(max_length=200)


class NotAModel:
    """Bare class — must not appear in the index."""
    title = "ignored"
