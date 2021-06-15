from django.shortcuts import render
from django.http import HttpResponse
import os
from dotenv import load_dotenv
from itertools import groupby
from statsmodels.stats.inter_rater import fleiss_kappa
import requests
import pandas as pd

# Create your views here.
def index(request):
    comments = []
    page = 1
    load_dotenv()

    while True:
        r = requests.get(f'https://api.github.com/repos/bhermann/DoR/issues/comments?page={page}&per_page=100', data={
            'Authorization': 'token ' + os.getenv('TOKEN')
        })
        if r.status_code != 200:
            return render(request, 'error.html', {'status': r.status_code})

        cur_comments = r.json()
        page += 1
        if len(cur_comments) == 0:
            break
        
        comments.extend(cur_comments)

    # Group comments by issue
    comments.sort(key=lambda p: p['issue_url'])
    groups = groupby(comments, key=lambda p: p['issue_url'])
    groups = [(key.split('/')[-1], list(data)) for key, data in groups]

    final_groups = []

    # Process each issue
    for key, data in groups:
        data = [x for x in data if 'reused:' in x['body']]

        # If there are no comments about reuse, skip
        if len(data) == 0:
            continue

        cur_groups = {}

        # Process each comment
        for comment in data:
            body = comment['body']

            # Process each document
            for line in body.split('\n'):
                if 'reused' in line:
                    # Mine the document and the reused artifacts
                    document = line.split('reused:')[0].split()[-1]
                    reused = line.split('reused:')[1].split(',')
                    reused = [x.strip() for x in reused]

                    # If document not in groups, add it
                    if document not in cur_groups:
                        cur_groups[document] = pd.DataFrame(
                            index=reused, columns=['y', 'n'])

                    # Process each artifact
                    for artifact in reused:
                        if artifact not in cur_groups[document].index:
                            # Seed the index
                            cur_groups[document].index.append(artifact)
                            cur_groups[document].loc[artifact, 'y'] = 2
                            cur_groups[document].loc[artifact,
                                                     'n'] = len(data) - 1

                        # Populate the DataFrame
                        if cur_groups[document].loc[artifact, :].sum() != len(data):
                            cur_groups[document].loc[artifact, 'y'] = 2
                            cur_groups[document].loc[artifact,
                                                     'n'] = len(data) - 1
                        else:
                            cur_groups[document].loc[artifact, 'y'] += 2
                            cur_groups[document].loc[artifact, 'n'] -= 2

        try:
            final_groups.append(
                (key, [(k, fleiss_kappa(df.to_numpy(), 'uniform')) for k, df in cur_groups.items()]))
        except:
            pass

    return render(request, "index.html", {'groups': final_groups})
