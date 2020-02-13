from django.urls import path

from . import views

app_name = 'stock'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('vote/', views.vote, name='vote'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
]
