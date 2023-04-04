from django.shortcuts import render

# Create your views here.
from datetime import datetime
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render
from webapp.models import BlogPost, BlogPostForm, CategoryForm, PageForm, CategoryModel, PageModel, EmpForm, EmpModel, AuthorForm, AuthorModel, PublisherForm, PublisherModel,BookForm,BookModel, ProductForm, ProductModel


def new_post(request):
    posts = BlogPost.objects.all()
    return render(request, 'newpost.html', {
        'posts': posts,
        'form': BlogPostForm()
    })


def create(request):
    if request.method == 'POST':
        form = BlogPostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.timestamp = datetime.now()
            post.save()
    return HttpResponseRedirect('/webapp/newpost')


def q1_1(request):
    categories = CategoryModel.objects.all()
    pages = PageModel.objects.all()
    return render(
        request, 'q1.html', {
            'categories': categories,
            'pages': pages,
            'cat_form': CategoryForm(),
            'page_form': PageForm()
        })


def q1_2(req):
    if req.method == 'POST':
        cat_form = CategoryForm(req.POST)
        page_form = PageForm(req.POST)

        if cat_form.is_valid() and page_form.is_valid():
            cat = cat_form.save(commit=False)
            page = page_form.save(commit=False)

            if CategoryModel.objects.last():
                cat.index = CategoryModel.objects.last().index + 1
                page.index = PageModel.objects.last().index + 1
            else:
                cat.index = 1
                page.index = 1

            cat.save()
            page.save()

    return HttpResponseRedirect('/webapp/q1_1')


def q2_1(req):
    emps = EmpModel.objects.all()
    return render(req, 'q2_1.html', {'emps': emps, 'form': EmpForm()})


def q2_2(req):
    if req.method == 'POST':
        form = EmpForm(req.POST)

        if form.is_valid():
            obj = form.save(commit=False)
            obj.save()

    return HttpResponseRedirect('/webapp/q2_1')


def q2_3(req):
    if req.method == 'POST':
        company = req.POST['company']

        if company:
            emps = EmpModel.objects.filter(company=company)
            return render(req, 'q2_2.html', {'emps': emps})

    return HttpResponse("<h1>Incorrect Response</h1>")


# Create your views here.
def q3_1(req):
    authors = AuthorModel.objects.all()
    publishers = PublisherModel.objects.all()
    books = BookModel.objects.all()

    return render(
        req, 'q3_1.html', {
            'authors': authors,
            'author_form': AuthorForm(),
            'publishers': publishers,
            'publisher_form': PublisherForm(),
            'books': books,
            'book_form': BookForm()
        })


def q3_2(req):
    if req.method == 'POST':
        author_form = AuthorForm(req.POST)
        publisher_form = PublisherForm(req.POST)

        if author_form.is_valid() and publisher_form.is_valid(
        ):
            author = author_form.save(commit=False)
            publisher = publisher_form.save(commit=False)

            author.save()
            publisher.save()

    return HttpResponseRedirect('/webapp/q3_1/')


def q3_3(req):
    if req.method == 'POST':
        book_form = BookForm(req.POST)
        if book_form.is_valid():
            book = book_form.save(commit=False)
            book.date = datetime.now()
            book.save()

    return HttpResponseRedirect('/webapp/q3_1/')

def q4_1(req):
    prods = ProductModel.objects.all()

    return render(req, 'q4_1.html', {'prods': prods, 'prod_form': ProductForm()})


def q4_2(req):
    if req.method == 'POST':
        prod_form = ProductForm(req.POST)

        print("=== Reached Here ===")

        if prod_form.is_valid():
            prod = prod_form.save(commit=False)
            prod.save()
        else:
            print("ERROR")

    return HttpResponseRedirect('/webapp/q4_1/')