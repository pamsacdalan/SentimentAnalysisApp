import re
import string
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import joblib
from joblib import load
# from nltk.stem import WordNetLemmatizer
# import nltk


device = "cpu"

df = pd.read_csv('twitter_cleaned.csv')
df = df.dropna()

def to_lower(message):
    result = message.lower()
    return result

def remove_num(message):
    result = re.sub(r'\d+','',message)
    return result

def contractions(message):
     result = re.sub(r"won't", "will not",message)
     result = re.sub(r"would't", "would not",message)
     result = re.sub(r"could't", "could not",message)
     result = re.sub(r"\'d", " would",message)
     result = re.sub(r"can\'t", "can not",message)
     result = re.sub(r"n\'t", " not", message)
     result = re.sub(r"\'re", " are", message)
     result = re.sub(r"\'s", " is", message)
     result = re.sub(r"\'ll", " will", message)
     result = re.sub(r"\'t", " not", message)
     result = re.sub(r"\'ve", " have", message)
     result = re.sub(r"\'m", " am", message)
     return result
    
def remove_punctuation(message):
    result = message.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_whitespace(message):
    result = message.strip()
    result = re.sub(' +',' ',message)
    return result

def replace_newline(message):
    result = message.replace('\n','')
    return result

def data_cleanup(message):
    cleaning_utils = [to_lower, remove_num, contractions, remove_punctuation, remove_whitespace, replace_newline]
    for util in cleaning_utils:
        message = util(message)
    return message


x_train, x_test, y_train, y_test = train_test_split(
    df['message'], df['category'], test_size=.2, stratify=df['label'])


# Building Model
vectorizer = TfidfVectorizer(max_features=2000)
# vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')

# Learn vocabulary from training texts and vectorize training texts.
x_train = vectorizer.fit_transform(x_train)

# Vectorize test texts.
x_test = vectorizer.transform(x_test)

x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()

y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)


def topk_encoding(nd_array):
    """
    Function to flatten the predicted category
    """
    
    predictions = nd_array
    
    ps = torch.exp(predictions)
    top_p, top_class  = ps.topk(1, dim=1)
    

    return top_class

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.hidden_layer_1 = nn.Linear(x_train.shape[1], 64) # input to first hidden layer
        self.output_layer = nn.Linear(64, self.out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        y = self.output_layer(x)
        y = self.activation(y)
        y = self.softmax(y)
        
        return y
    
model = NeuralNetwork(x_train.shape[1], df['category'].nunique())
# model = NeuralNetwork(x_train.shape[1], 5)


# Define the loss
criterion = nn.NLLLoss()


# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.002)

#setting up scheduler
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 10)


# Model Fine-Tuning
train_losses = []
test_losses = []
test_accuracies = []

epochs = 200
for e in range(epochs):
    optimizer.zero_grad()

    output = model.forward(x_train) #Forward pass, get the logits
    loss = criterion(output, y_train) # Calculate the loss with the logits and the labels
    loss.backward()
    train_loss = loss.item()
    train_losses.append(train_loss)
    
    optimizer.step()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        log_ps = model.forward(x_test)
        test_loss = criterion(log_ps, y_test)
        test_losses.append(test_loss)

        ps = torch.exp(log_ps)
        top_p, top_class  = ps.topk(1, dim=1)
        equals = top_class == y_test.view(*top_class.shape)
        test_accuracy = torch.mean(equals.float())
        test_accuracies.append(test_accuracy)

    model.train()

#     scheduler.step(test_loss/len(y_test))


def input_vectorizer(message):
    """
    Function to predict the category of inputted message
    """
    
    cleaned_message = pd.Series(message).apply(lambda x: data_cleanup(x))
    vec = vectorizer.transform(pd.Series(cleaned_message))
    vec = torch.tensor(scipy.sparse.csr_matrix.todense(vec)).float()
    preds = model.forward(vec)
    category = topk_encoding(preds).detach().cpu().numpy()
    
    return int(category[0])