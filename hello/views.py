from django.shortcuts import render
from django.http import HttpResponse
import os
from dotenv import load_dotenv
from itertools import groupby
from pandas.core.algorithms import isin
from pandas.core.indexes import multi
import requests
from statsmodels.stats.inter_rater import fleiss_kappa
import requests_cache
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import base64
import io
from io import StringIO
import matplotlib as mpl
import re
from pprint import pprint

FILE_REGEX = '\[.*\]\((https:\/\/github.com\/bhermann\/DoR\/files\/.*)\)'
sns.set()

def normalize_index(x):
    if not isinstance(x, str):
        if not math.isnan(x):
            x = str(int(x))
        else:
            x = str(x)
                                
    x = x.strip()
    if x[0] != '[':
        x = f'[{x}'
    if x[-1] != ']':
        x =  f'{x}]'
    return x


# Create your views here.
def index(request):
    comments = []
    page = 1
    load_dotenv()

    while True:
        print('Fetching page', page)
        session = requests_cache.CachedSession('dor-cache', expire_after=3600)
        r = session.get(f'https://api.github.com/repos/bhermann/DoR/issues/comments?page={page}&per_page=100', data={
            'Authorization': 'token ' + os.getenv('TOKEN')
        })
        if r.status_code != 200:
            print(r.content)
            return render(request, 'error.html', {'status': r.status_code})

        cur_comments = r.json()
        page += 1
        if len(cur_comments) == 0:
            break
        
        comments.extend(cur_comments)
    
    print('Processing', len(comments), 'comments.')

    # Group comments by issue
    comments.sort(key=lambda p: p['issue_url'])
    groups = groupby(comments, key=lambda p: p['issue_url'])
    groups = [(key.split('/')[-1], list(data)) for key, data in groups]

    final_groups = []
    scores_only = []

    # Process each issue
    for key, data in groups:
        # Filter data
        data = [x for x in data if 'reused:' in x['body'] or len(re.findall(FILE_REGEX, x['body'])) > 0]

        # If there are no comments about reuse, skip
        if len(data) == 0:
            continue

        cur_groups = {}

        # Are there multiple comments?
        multiple_comments = (len(data) > 1)

        # Process each comment
        for comment in data:
            body = comment['body']

            # Check for file in the comment.
            link = re.findall(FILE_REGEX, body)
            if len(link) > 0:
                # We have a comment with a CSV
                url = link[0]

                # Get the file
                with requests.get(url) as r:
                    r.raise_for_status()

                    try:
                        df: pd.DataFrame = pd.read_csv(StringIO(r.content.decode('utf-8')), sep=None, engine='python')
                    except pd.errors.ParserError as err:
                        print('Parse error in issue', key)
                        print()
                        continue

                    df.columns = [x.strip() for x in df.columns]

                    try:
                        df.dropna(axis=0, inplace=True, subset=['paper_doi'])
                    except KeyError as err:
                        print(err, df.columns)
                        
                    try:
                        df['paper_doi'] = [x.strip() for x in df['paper_doi']]
                    except KeyError as err:
                        print('In issue', key, 'column paper_doi not found. Columns are', df.columns)
                        print()
                        continue
                    except AttributeError as err:
                        print(err, df['paper_doi'])
                        continue

                    groups = df.groupby('paper_doi')
                    for doi, group in groups:
                        if doi not in cur_groups:
                            # Build the DataFrame index
                            index = group['citation_number']
                            index = set([normalize_index(x) for x in index])

                            cur_groups[doi] = pd.DataFrame(
                                index=index, columns=['y', 'n'])
                        
                        for artifact in group['citation_number']:
                            artifact = normalize_index(artifact)
                            if artifact not in cur_groups[doi].index:
                                # Seed the index
                                cur_groups[doi].loc[artifact, 'y'] = (2 - multiple_comments)
                                cur_groups[doi].loc[artifact,
                                                        'n'] = len(data) - 1
                            
                            # Populate the DataFrame
                            if cur_groups[doi].loc[artifact, :].sum() != len(data):
                                cur_groups[doi].loc[artifact, 'y'] = (2 - multiple_comments)
                                cur_groups[doi].loc[artifact,
                                                        'n'] = len(data) - 1
                            else:
                                cur_groups[doi].loc[artifact, 'y'] += (2 - multiple_comments)
                                cur_groups[doi].loc[artifact, 'n'] -= (2 - multiple_comments)

                continue
            elif 'reused:' not in body:
                continue
            # Process each document
            for line in body.split('\n'):
                if 'reused' in line:
                    # Mine the document and the reused artifacts
                    document = line.split('reused:')[0].split()[-1].strip()
                    reused = line.split('reused:')[1].split(',')
                    reused = [normalize_index(x) for x in reused]

                    # If document not in groups, add it
                    if document not in cur_groups:
                        cur_groups[document] = pd.DataFrame(
                            index=reused, columns=['y', 'n'])

                    # Process each artifact
                    for artifact in reused:
                        artifact = normalize_index(artifact)
                        if artifact not in cur_groups[document].index:
                            # Seed the index
                            cur_groups[document].loc[artifact, 'y'] = (2 - multiple_comments)
                            cur_groups[document].loc[artifact,
                                                     'n'] = len(data) - 1

                        # Populate the DataFrame
                        if cur_groups[document].loc[artifact, :].sum() != len(data):
                            cur_groups[document].loc[artifact, 'y'] = (2 - multiple_comments)
                            cur_groups[document].loc[artifact,
                                                     'n'] = len(data) - 1
                        else:
                            cur_groups[document].loc[artifact, 'y'] += (2 - multiple_comments)
                            cur_groups[document].loc[artifact, 'n'] -= (2 - multiple_comments)

        try:
            if len(cur_groups) > 0:
                kappas = []
                for k, df in cur_groups.items():
                    kappa = min(1., round(fleiss_kappa(df.to_numpy(), 'uniform'), 2))
                    kappas.append((k, kappa))

                    if multiple_comments:
                        scores_only.append(kappa)
                final_groups.append((key, kappas))
        except:
            pass

    _, ax = plt.subplots()
    sns.histplot(data=scores_only, alpha=0.7, kde=True,  ax=ax)
    ax.set_xlabel('Fleiss\' Kappa')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Kappa scores')

    # Plot the minor gridlines too
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    base64_data = base64.b64encode(buf.read())
    base64_data = 'data:image/jpg;base64,' + base64_data.decode('utf-8')

    num_papers = len(scores_only)

    return render(request, "index.html", {'num_papers': num_papers, 'groups': final_groups, 'hist': base64_data})
