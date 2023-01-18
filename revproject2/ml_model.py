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
# from nltk.stem import WordNetLemmatizer
# import nltk


device = "cpu"

df = pd.read_csv('twitter_cleaned.csv')
df = df.dropna()

x_train, x_test, y_train, y_test = train_test_split(
    df['message'], df['category'], test_size=.2, stratify=df['label'], random_state=0)

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


preds = model.forward(x_test)
preds = topk_encoding(preds)

def input_vectorizer(message):
    """
    Function to predict the category of inputted message
    """
    
    vec = vectorizer.transform(pd.Series(message))
    vec = torch.tensor(scipy.sparse.csr_matrix.todense(vec)).float()
    preds = model.forward(vec)
    category = topk_encoding(preds).detach().cpu().numpy()
    
    return int(category[0])
