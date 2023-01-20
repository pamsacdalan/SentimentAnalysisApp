
from django.shortcuts import render
from .forms import SentimentsForm
import re
import string
import joblib
from joblib import load
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_model import input_vectorizer
import torch
from torch import nn
import scipy
import numpy


#model = joblib.load('D:/77Global Training\Revalida 2\Revalida2-project/tfidf_vectorizer.joblib')
#vectorizer = TfidfVectorizer(max_features=2000)

# Create your views here.
def home(request):
    return render(request, 'home.html')

def getPrediction(request):
    
    """ This function handles the review input and prediction of result """
    
    
    form = SentimentsForm(request.POST)
    context = {}

    
    if request.method == 'POST' :
        
        if form.is_valid():
            final_review = form['review'].value() #extracts the review to be analyzed'
            print(final_review)
            prediction = input_vectorizer(final_review)
            #print(prediction)
            context = {'result': prediction, 'input': final_review}
            #context['text'] = prediction
            #print(context)

            # 0 - negative // 1 - Neutral // 2 - Positive

    else:
        form = SentimentsForm()

    
    context['form'] = form
    return render(request, 'analyze.html', context=context)



