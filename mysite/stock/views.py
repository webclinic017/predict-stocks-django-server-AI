from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic

from stock import predict_stock
import numpy as np


class IndexView(generic.ListView):
    template_name = 'polls/index.html'

    def get_queryset(self):
        return None


class ResultsView(generic.DetailView):
    template_name = 'polls/results.html'

#getting request, making prediction, returning values
def vote(request):
    if (request.method == 'POST'):
        #getting request
        stock_code = request.POST.get('stock_code')
        days_to_predict = request.POST.get('days_to_predict')
        print(stock_code)
        #predicting
        x, y, a, b, c = predict_stock(stock_code, days_to_predict)

        print(len(x), len(c))

        #checking if inputs are valid
        if x == "error":
            return render(request, 'polls/index.html', {'error_message': True})

        #returning request
        return render(request, 'polls/results.html', {'Predictions': x, 'Prices': y, 'Volume_Predictions': a, 'Volumes': b, "Dates": c, "Stock_Name": stock_code})
