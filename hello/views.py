from django.shortcuts import render
import os
from dotenv import load_dotenv
from itertools import groupby
from statsmodels.stats.inter_rater import fleiss_kappa
import requests_cache
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import base64
import io
import statistics
from io import StringIO
import matplotlib as mpl
import re
from pprint import pprint
from enum import Enum
import time


class IgnoreDirectives(Enum):
    """
    Specifies a set of comment parts that we should ignore.
    """
    LINKS = 1
    TIMES = 2


# Are we debugging?
DEBUG = False

FILE_REGEX = '\[.*\]\((https:\/\/github.com\/bhermann\/DoR\/files\/.*)\)'

# Regex for time. Matches the formats:
# 24:09
# 24m 5s --> does not match the "s", but we do not need it anyway
TIME_REGEX = '\d+m?(?:\s+)?:\d+'

sns.set()


def normalize_index(x):
    """
    Normalizes a citation. Adds in square brackets on each side
    and removes spacing on the sides. Also converts to a str.
    """
    if not isinstance(x, str):
        if not math.isnan(x):
            x = str(int(x))
        else:
            x = str(x)

    x = x.strip()

    if len(x) == 0:
        return x
    if x[0] != '[':
        x = f'[{x}'
    if x[-1] != ']':
        x = f'{x}]'
    return x


def normalize_doi(x):
    """
    Normalizes the DOIs to a URL.
    """
    x = x.strip()
    if not x.startswith('http'):
        x = f'https://doi.org/{x}'

    return x


# Create your views here.
def index(request):
    comments = []
    page = 1
    load_dotenv()

    start = time.time()

    while True:
        session = requests_cache.CachedSession(
            'dor-cache', expire_after=3600 * 24)
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

    if DEBUG:
        print('Processing', len(comments), 'comments.', round(
            time.time() - start, 2), 'seconds elapsed.')

    # Group comments by issue
    comments.sort(key=lambda p: p['issue_url'])
    groups = groupby(comments, key=lambda p: p['issue_url'])
    groups = [(key.split('/')[-1], list(data)) for key, data in groups]

    # For kappa stats
    final_groups = []
    scores_only = []

    # How long does it take to read papers?
    reading_times = []

    # How many issues do we have?
    num_issues = len(groups)

    # The other stats
    num_papers = {
        'multiple_reviewers': 0,
        'all_papers': 0,
        'kappa_1': 0,
        'good_kappa': 0
    }

    # For debug purposes
    issues_parsing_start_time = time.time()
    issues_parsing_times = []

    # Process each issue
    for key, data in groups:
        # Filter data
        data = [x for x in data if 'reused:' in x['body']
                or len(re.findall(FILE_REGEX, x['body'])) > 0]

        # If there are no comments about reuse, skip
        if len(data) == 0:
            continue

        cur_groups = {}

        # Are there multiple comments?
        multiple_comments = (len(data) > 1)

        # Process each comment
        for comment in data:
            body = comment['body']

            # Should we be ignoring anything?
            ignore_directives = []
            if body.startswith('#ignore'):
                # What should we ignore?
                first_line = body.split('\n')[0].split()[1:]
                for item in first_line:
                    if item == 'links':
                        ignore_directives.append(IgnoreDirectives.LINKS)
                    if item == 'times':
                        ignore_directives.append(IgnoreDirectives.TIMES)

            # Parse times (if we should)
            if IgnoreDirectives.TIMES not in ignore_directives:
                # Get lines with the times
                comment_lines = body.split('\n')
                comment_lines = [x for x in comment_lines if ' took' in x]

                # Grab those times
                for line in comment_lines:
                    # Look for times
                    intervals = re.findall(TIME_REGEX, line)
                    if len(intervals) != 0:
                        interval = intervals[0]
                        if ' ' in interval:
                            # We're processing a case of the format "24m 5s"
                            interval_split = interval.split()

                            try:
                                minutes = float(interval_split[0][:-1])
                                minutes += float(interval_split[1]) / 60
                                reading_times.append(minutes)
                            except ValueError as err:
                                print('Parsing error occurred for timestamp:', err)
                                continue
                        else:
                            # We're parsing a case of the format mm:ss
                            interval_split = interval.split(':')

                            try:
                                minutes = float(interval_split[0])
                                minutes += float(interval_split[1]) / 60
                                reading_times.append(minutes)
                            except ValueError as err:
                                print('Parsing error occurred for timestamp:', err)
                                continue

            # Make sure we want to parse links
            if IgnoreDirectives.LINKS not in ignore_directives:
                # Check for file in the comment.
                link = re.findall(FILE_REGEX, body)
                if len(link) > 0:
                    # We have a comment with a CSV
                    url = link[0]

                    # Get the file
                    session = requests_cache.CachedSession(
                        'dor-cache-files', expire_after=3600*24*7)
                    with session.get(url) as r:
                        r.raise_for_status()

                        # Get the data
                        try:
                            df: pd.DataFrame = pd.read_csv(
                                StringIO(r.content.decode('latin-1')), sep=None, engine='python')
                        except:
                            # We can get a _csv.Error or ParserError, but we cannot catch
                            # _csv.Error apparently.
                            print('Parse error in issue', key)
                            print()
                            continue

                        # Preprocess the columns
                        df.columns = [x.strip().lower().replace(' ', '_').replace('citation_no', 'citation_number') for x in df.columns]

                        # Dump rows with no paper_doi (reusing DOI)
                        try:
                            df.dropna(axis=0, inplace=True,
                                      subset=['paper_doi'])
                        except KeyError as err:
                            print(err, df.columns)

                        # Normalize the DOIs
                        try:
                            df['paper_doi'] = [normalize_doi(
                                x) for x in df['paper_doi']]
                        except KeyError as err:
                            print(
                                'In issue', key, 'column paper_doi not found. Columns are', df.columns)
                            print()
                            continue
                        except AttributeError as err:
                            print(err, df['paper_doi'])
                            continue

                        # For each paper, check agreement
                        groups = df.groupby('paper_doi')
                        for doi, group in groups:
                            if doi not in cur_groups:
                                # Build the DataFrame index
                                index = group['citation_number']
                                index = list(set([normalize_index(x)
                                             for x in index]))

                                cur_groups[doi] = pd.DataFrame(
                                    index=index, columns=['y', 'n'])

                            for artifact in group['citation_number']:
                                artifact = normalize_index(artifact)
                                if artifact not in cur_groups[doi].index:
                                    # Seed the index
                                    cur_groups[doi].loc[artifact, 'y'] = (
                                        2 - multiple_comments)
                                    cur_groups[doi].loc[artifact,
                                                        'n'] = len(data) - 1

                                # Populate the DataFrame
                                if cur_groups[doi].loc[artifact, :].sum() != len(data):
                                    cur_groups[doi].loc[artifact, 'y'] = (
                                        2 - multiple_comments)
                                    cur_groups[doi].loc[artifact,
                                                        'n'] = len(data) - 1
                                else:
                                    cur_groups[doi].loc[artifact,
                                                        'y'] += (2 - multiple_comments)
                                    cur_groups[doi].loc[artifact,
                                                        'n'] -= (2 - multiple_comments)

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
                                cur_groups[document].loc[artifact, 'y'] = (
                                    2 - multiple_comments)
                                cur_groups[document].loc[artifact,
                                                         'n'] = len(data) - 1

                            # Populate the DataFrame
                            if cur_groups[document].loc[artifact, :].sum() != len(data):
                                cur_groups[document].loc[artifact, 'y'] = (
                                    2 - multiple_comments)
                                cur_groups[document].loc[artifact,
                                                         'n'] = len(data) - 1
                            else:
                                cur_groups[document].loc[artifact,
                                                         'y'] += (2 - multiple_comments)
                                cur_groups[document].loc[artifact,
                                                         'n'] -= (2 - multiple_comments)

        try:
            # Do we have kappas in the issue?
            if len(cur_groups) > 0:
                kappas = []
                for k, df in cur_groups.items():
                    # Workaround for issue with kappa > 1
                    kappa = min(1., round(fleiss_kappa(
                        df.to_numpy(), 'uniform'), 2))
                    kappas.append((k, kappa))

                    # Collect stats
                    num_papers['all_papers'] += 1

                    if multiple_comments:
                        scores_only.append(kappa)
                        num_papers['multiple_reviewers'] += 1

                        if kappa == 1:
                            num_papers['kappa_1'] += 1

                        if kappa > 0.6:
                            num_papers['good_kappa'] += 1
                final_groups.append((key, kappas))
        except:
            pass

        # Record the time taken to process the issue
        issues_parsing_times.append((key, len(data), round(
            time.time() - issues_parsing_start_time, 2)))
        issues_parsing_start_time = time.time()

    _, [ax0, ax1] = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 8), dpi=150, tight_layout=True)

    sns.histplot(data=scores_only, alpha=0.7, kde=True,  ax=ax0)
    ax0.set_xlabel('Fleiss\' Kappa')
    ax0.set_ylabel('Frequency')
    ax0.set_title('Distribution of Kappa scores')

    print(statistics.median(scores_only), statistics.mean(scores_only))

    # Plot the minor gridlines too
    ax0.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax0.grid(b=True, which='major', color='w', linewidth=1.0)
    ax0.grid(b=True, which='minor', color='w', linewidth=0.5)

    # Now plot times
    sns.histplot(data=reading_times, alpha=0.7,
                 kde=True, stat='probability', ax=ax1)
    ax1.set_xlabel('Reading time (min)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Distribution of reading times')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    base64_data = base64.b64encode(buf.read())
    base64_data = 'data:image/jpg;base64,' + base64_data.decode('utf-8')

    # Print the issue parsing times
    if DEBUG:
        print('Total time elapsed:', time.time() - start, 'seconds')
        pprint(issues_parsing_times)

    return render(request, "index.html",
                  {
                      'num_papers': num_papers,
                      'num_issues': num_issues,
                      'groups': final_groups,
                      'hist': base64_data,
                      'median_read_time': round(statistics.median(reading_times), 2)
                  }
                  )
