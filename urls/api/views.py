from django.shortcuts import render,redirect
from django.http import HttpResponse
from django import forms
from backend.api.preprocessing import preprocessing
from backend.api.predict import execute
import time
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
class AnswerForm(forms.Form):
   answer = forms.CharField(max_length = 100)
   

# Input Page
def input(request):
    if request.method == "POST":

        answer = AnswerForm(request.POST)
        if answer.is_valid():
            answer = answer.cleaned_data['answer']

        # processing of answer 3 methods 
        #1 preprocessing_bow
        #2 preprocessing_bow_features
        #3 preprocessing  ( only using features )
        start_time = time.time()   
        preprocessed_answer= preprocessing(answer,isTest=True)
        #print(preprocessed_answer)

        #################Use this answer vector and directly predict################



        marks = execute(preprocessed_answer)















        ################################################################################


        print("Total Time taken for all three models",time.time()-start_time)
        return render(request,"Result.html",{'vectoranswer':preprocessed_answer,'prediction':marks})
    return render(request,"form.html")

##  Result Page    
# def results(request):
#     return render(request,"Result.html")

    
