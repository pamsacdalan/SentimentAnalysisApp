
from django.shortcuts import render
from .forms import SentimentsForm
import joblib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ml_model
from ml_model import topk_encoding
import torch
import scipy
import numpy



# importing ml model here
if __name__=='__main__':
    with open('nlp_model.pkl', 'rb') as f:
        model = pickle.load(f)

if __name__=='__main__':
    with open('vectorizer.pkl', 'rb') as v:
        loaded_vectorizer = pickle.load(v)


#model = joblib.load("nlp_model.pkl")

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
            vec = loaded_vectorizer.transform(final_review).toarray() #transform the review
            vec = torch.tensor(scipy.sparse.csr_matrix.todense(vec)).float()
            prediction = model.forward(vec)
            category = topk_encoding(prediction).detach().cpu().numpy()
            context['text'] = category
            print(context['text'])
    
    else:
        form = SentimentsForm()

    
    context['form'] = form
    return render(request, 'analyze.html', context=context)



