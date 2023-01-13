from django import forms
from .models import SentimentModel

class SentimentsForm(forms.Form):
    review = forms.CharField(max_length=120, widget=forms.Textarea(
        attrs = {
            'class' : 'form-control',
            'placeholder' : 'Input comment..',
        }
    ))
    
    class Meta:
        model = SentimentModel
        fields = [
            'Enter sentence to analyze'
        ]
    