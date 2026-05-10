from django.db import models


class Author(models.Model):
    name = models.CharField(max_length=200)


class Article(models.Model):
    author = models.ForeignKey(
        Author,
        on_delete=models.CASCADE,
        related_name="articles",
    )
    title = models.CharField(max_length=200)


class Comment(models.Model):
    # Default reverse name -> "comment_set" on Article
    article = models.ForeignKey("blog.Article", on_delete=models.CASCADE)
    body = models.TextField()


class Tag(models.Model):
    name = models.CharField(max_length=50)
    # M2M with explicit related_name
    articles = models.ManyToManyField(Article, related_name="tags")


class HiddenLink(models.Model):
    # related_name="+" disables the reverse
    article = models.ForeignKey(
        Article, on_delete=models.CASCADE, related_name="+"
    )
