from django.db import models

# Create your models here.
from django import forms


class BlogPost(models.Model):
    title = models.CharField(max_length=150)
    body = models.TextField()
    timestamp = models.DateTimeField()


#  class Meta:
#  	ordering = ('-timestamp',)


class BlogPostForm(forms.ModelForm):

    class Meta:
        model = BlogPost
        exclude = ('timestamp', )


class CategoryModel(models.Model):
    index = models.PositiveIntegerField()
    name = models.CharField(max_length=100)
    visits = models.PositiveIntegerField()
    likes = models.PositiveIntegerField()


class PageModel(models.Model):
    index = models.PositiveIntegerField()
    category = models.CharField(max_length=100)
    title = models.CharField(max_length=100)
    url = models.URLField()
    views = models.PositiveIntegerField()


class CategoryForm(forms.ModelForm):

    class Meta:
        model = CategoryModel
        exclude = ('index', )


class PageForm(forms.ModelForm):

    class Meta:
        model = PageModel
        exclude = ('index', )


class EmpModel(models.Model):
    name = models.CharField(max_length=100, primary_key=True)
    company = models.CharField(max_length=30)
    salary = models.PositiveIntegerField()
    street = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

    class Meta:
        ordering = ('name', )


class EmpForm(forms.ModelForm):

    class Meta:
        model = EmpModel
        exclude = ()


class AuthorModel(models.Model):
    fname = models.CharField(max_length=50, unique=True)
    lname = models.CharField(max_length=50)
    email = models.EmailField()

    class Meta:
        ordering = ("fname", )


# publisher model
class PublisherModel(models.Model):
    name = models.CharField(max_length=100, unique=True)
    street = models.CharField(max_length=20)
    city = models.CharField(max_length=20)
    state = models.CharField(max_length=20)
    country = models.CharField(max_length=20)
    url = models.URLField()

    class Meta:
        ordering = ("name", )


# book
class BookModel(models.Model):
    title = models.CharField(max_length=100)
    date = models.DateTimeField()
    authors = models.ForeignKey(AuthorModel,
                                on_delete=models.CASCADE,
                                to_field='fname')
    publisher = models.ForeignKey(PublisherModel,
                                  on_delete=models.CASCADE,
                                  to_field='name')

    class Meta:
        ordering = ("title", )


class AuthorForm(forms.ModelForm):

    class Meta:
        model = AuthorModel
        exclude = ()


class PublisherForm(forms.ModelForm):

    class Meta:
        model = PublisherModel
        exclude = ()


class BookForm(forms.ModelForm):
    class Meta:
        model = BookModel
        exclude = ('date',)


class ProductModel(models.Model):
    title = models.CharField(max_length=100)
    price = models.PositiveIntegerField()
    des = models.CharField(max_length=1000)

class ProductForm(forms.ModelForm):
    class Meta:
        model = ProductModel
        exclude = ('',)