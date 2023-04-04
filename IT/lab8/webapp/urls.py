from django.conf.urls import include, url
from . import views

urlpatterns = [
    url('newpost/', views.new_post),
    url('create/', views.create),
    url('q1_1/', views.q1_1),
    url('q1_2/', views.q1_2),
    url('q2_1/', views.q2_1),
    url('q2_2/', views.q2_2),
    url('q2_3/', views.q2_3),
    url('q3_1/', views.q3_1),
    url('q3_2/', views.q3_2),
    url('q3_3/', views.q3_3),
    url('q4_1/', views.q4_1),
    url('q4_2/', views.q4_2),

]