
from django.contrib import admin
from webapp.models import BlogPost,CategoryModel,PageModel,EmpModel, AuthorModel, PublisherModel, BookModel, ProductModel

admin.site.register(BlogPost)
admin.site.register(PageModel)
admin.site.register(CategoryModel)
admin.site.register(EmpModel)
admin.site.register(AuthorModel)
admin.site.register(PublisherModel)
admin.site.register(BookModel)
admin.site.register(ProductModel)


