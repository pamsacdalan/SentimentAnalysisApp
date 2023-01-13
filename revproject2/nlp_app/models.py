from django.db import models

# Create your models here.
class SentimentModel(models.Model):
    review = models.CharField(max_length=120)