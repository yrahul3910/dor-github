from django.shortcuts import render
from django.http import HttpResponse
import os
from dotenv import load_dotenv
import requests

from .models import Greeting

# Create your views here.


def index(request):
    load_dotenv()
    r = requests.get('https://api.github.com/repos/bhermann/DoR/issues/comments', data={
        'Authorization': 'token ' + os.getenv('TOKEN')
    })
    return render(request, "index.html", {'issues': r.json()})
