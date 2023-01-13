
from django.shortcuts import render
from .forms import SentimentsForm
#import 'training model (code)' here

# Create your views here.

def home(request):
    return render(request, 'home.html')

def input_review(request):
    
    """ This function handles the review input and prediction of result """
    
    form = SentimentsForm(request.POST or None)
    context = {}
    
    if request.method == 'POST' :
        if form.is_valid():
            final_review = form.cleaned_data.get('review') #extracts the review to be analyzed
            #prediction = model(final_review)  -- insert model here
            #context['text'] = prediction
    
    else:
        form = SentimentsForm()
    

    context['form'] = form   
    return render(request, 'analyze.html', context=context)



