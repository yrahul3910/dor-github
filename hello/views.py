from django.shortcuts import render
from django.http import HttpResponse
import os
from dotenv import load_dotenv
from itertools import groupby
import requests

from .models import Greeting

# Create your views here.


def index(request):
    load_dotenv()
    r = requests.get('https://api.github.com/repos/bhermann/DoR/issues/comments', data={
        'Authorization': 'token ' + os.getenv('TOKEN')
    })

    comments = r.json()
    comments.sort(key=lambda p: p['issue_url'])
    groups = groupby(comments, key=lambda p: p['issue_url'])
    groups = [(key.split('/')[-1], list(data)) for key, data in groups]

    return render(request, "index.html", {'groups': groups})
