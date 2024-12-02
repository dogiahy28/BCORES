from flask import Flask, render_template, request, send_file, make_response, session, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import re
import hnswlib
from collections import defaultdict
import time

app = Flask(__name__)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
data_conference = pd.read_csv('data_conference.csv')
data_articles = pd.read_csv('data_article.csv')

data_conference.fillna('', inplace=True)
data_articles.fillna('', inplace=True)

def safe_json_to_array(x):
    if isinstance(x, str):
        try:
            return np.array(json.loads(x))
        except json.JSONDecodeError as e:
            print(f"Invalid string: {x}. Error: {e}")
            return np.array([])
    else:
        return np.array([])

data_conference['Title_Vector'] = data_conference['Title_Vector'].apply(safe_json_to_array)
data_conference['Description_Sent2vec'] = data_conference['Description_Sent2vec'].apply(safe_json_to_array)

valid_title_embeddings = data_conference['Title_Vector'].apply(lambda x: len(x) > 0)
valid_description_embeddings = data_conference['Description_Sent2vec'].apply(lambda x: len(x) > 0)

data_title_filtered = data_conference[valid_title_embeddings].reset_index(drop=True)
data_description_filtered = data_conference[valid_description_embeddings].reset_index(drop=True)

title_embeddings = np.vstack(data_title_filtered['Title_Vector'].values)
description_embeddings = np.vstack(data_description_filtered['Description_Sent2vec'].values)

dimension_title = title_embeddings.shape[1]
p_title = hnswlib.Index(space='cosine', dim=dimension_title)
p_title.init_index(max_elements=len(title_embeddings), ef_construction=300, M=20)
p_title.add_items(title_embeddings)
p_title.set_ef(50)

dimension_description = description_embeddings.shape[1]
p_description = hnswlib.Index(space='cosine', dim=dimension_description)
p_description.init_index(max_elements=len(description_embeddings), ef_construction=300, M=20)
p_description.add_items(description_embeddings)
p_description.set_ef(50)

data_articles['Title_Vector'] = data_articles['Title_Vector'].apply(safe_json_to_array)
data_articles['Abstract_Sent2vec'] = data_articles['Abstract_Sent2vec'].apply(safe_json_to_array)

valid_article_title_embeddings = data_articles['Title_Vector'].apply(lambda x: len(x) > 0)
valid_article_abstract_embeddings = data_articles['Abstract_Sent2vec'].apply(lambda x: len(x) > 0)

data_article_filtered = data_articles[valid_article_title_embeddings].reset_index(drop=True)

article_title_embeddings = np.vstack(data_article_filtered['Title_Vector'].values)
article_abstract_embeddings = np.vstack(data_article_filtered['Abstract_Sent2vec'].values)


dimension_article_title = article_title_embeddings.shape[1]
p_article_title = hnswlib.Index(space='cosine', dim=dimension_article_title)
p_article_title.init_index(max_elements=len(article_title_embeddings), ef_construction=200, M=20)
p_article_title.add_items(article_title_embeddings)
p_article_title.set_ef(50)

dimension_article_abstract = article_abstract_embeddings.shape[1]
p_article_abstract = hnswlib.Index(space='cosine', dim=dimension_article_abstract)
p_article_abstract.init_index(max_elements=len(article_abstract_embeddings), ef_construction=200, M=20)
p_article_abstract.add_items(article_abstract_embeddings)
p_article_abstract.set_ef(50)


def get_conference_by_id(article_id):
    conference_row = data_conference[data_conference['ERAID'] == article_id]

    if not conference_row.empty:
        return conference_row.iloc[0].to_dict()
    else:
        return None

@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
    articles = data_article_filtered.to_dict(orient='records')
    articles_per_page = 10
    total_articles = len(articles)
    total_pages = (total_articles + articles_per_page - 1) // articles_per_page
    start = (page - 1) * articles_per_page
    end = start + articles_per_page
    articles_to_display = articles[start:end]

    return render_template('index.html', articles=articles_to_display, page=page, total_pages=total_pages)

filtered_search_results = []

from collections import defaultdict
def highlight_keywords(text, keywords):
    excluded_words = {"and", "or", "the", "is", "are", "in", "of", "to", "a", "for", "on", "with"}
    long_keywords = [keyword for keyword in keywords if len(keyword) >= 2 and keyword.lower() not in excluded_words]

    for keyword in long_keywords:
        keyword_lower = keyword.lower()
        text = re.sub(rf'(?i)\b{keyword}\b', lambda m: f"<span style='background-color: #a9d8ff;'>{m.group(0)}</span>", text)
    return text


def read_weights_from_file():
    weights = {}
    with open('weights.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            weights[key] = float(value)
    return weights

def write_weights_to_file(weights):
    with open('weights.txt', 'w') as f:
        for key, value in weights.items():
            f.write(f"{key}:{value}\n")

CRITERIA_WEIGHTS = read_weights_from_file()

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global CRITERIA_WEIGHTS
    try:
        title_with_title_vector = float(request.form['weight_title_vector'])
        title_with_description_vector = float(request.form['weight_description_sent2vec'])
        abstract_with_title_vector = float(request.form['weight_abstract_title_vector'])
        abstract_with_description_vector = float(request.form['weight_abstract_description'])

        total_weight = title_with_title_vector + title_with_description_vector + abstract_with_title_vector + abstract_with_description_vector
        title_sum = title_with_title_vector + title_with_description_vector
        abstract_sum = abstract_with_title_vector + abstract_with_description_vector

        if total_weight != 1.0:
            flash("Error: Total of all weights must equal 100%.", "error")
            return render_template('admin_setting.html', weights=CRITERIA_WEIGHTS)

        # if title_sum != 0.5:
        #     flash("Error: Sum of Title weights must equal 50%.", "error")
        #     return render_template('admin_setting.html', weights=CRITERIA_WEIGHTS)

        # if abstract_sum != 0.5:
        #     flash("Error: Sum of Abstract weights must equal 50%.", "error")
        #     return render_template('admin_setting.html', weights=CRITERIA_WEIGHTS)

        CRITERIA_WEIGHTS['title_with_title_vector'] = title_with_title_vector
        CRITERIA_WEIGHTS['title_with_description_vector'] = title_with_description_vector
        CRITERIA_WEIGHTS['abstract_with_title_vector'] = abstract_with_title_vector
        CRITERIA_WEIGHTS['abstract_with_description_vector'] = abstract_with_description_vector

        write_weights_to_file(CRITERIA_WEIGHTS)

        flash("Weights updated successfully!", "success")
        return render_template('admin_setting.html', weights=CRITERIA_WEIGHTS)

    except ValueError:
        flash("Error: Please enter valid numeric values for weights.", "danger")
        return render_template('admin_setting.html', weights=CRITERIA_WEIGHTS)

@app.route('/recommend', methods=['POST'])
def recommend():
    title_input = request.form.get('title_input', "")
    ranks = request.form.get('ranks', "").split(',')
    selected_field = request.form.get('selected_field', "")
    k = int(request.form.get('num_results', 10))  # Nhận giá trị số lượng kết quả (mặc định là 10)
    action = request.form.get('action', "recommend")

    global filtered_search_results
    filtered_search_results = []

    start_time = time.time()
    input_embedding = model.encode([title_input]) if title_input else None

    N = 100
    borda_scores = defaultdict(float)

    weight_title = CRITERIA_WEIGHTS['title_with_title_vector']*2
    weight_description = CRITERIA_WEIGHTS['title_with_description_vector']*2

    if input_embedding is not None:
        labels_title_from_input, distances_title = p_title.knn_query(input_embedding, k=N)
        labels_description_from_input, distances_description = p_description.knn_query(input_embedding, k=N)

        ranking_title = {idx: rank + 1 for rank, idx in enumerate(labels_title_from_input[0])}
        ranking_description = {idx: rank + 1 for rank, idx in enumerate(labels_description_from_input[0])}

        max_points = N

        for idx, rank in ranking_title.items():
            borda_scores[idx] += weight_title * (max_points - rank + 1)

        for idx, rank in ranking_description.items():
            borda_scores[idx] += weight_description * (max_points - rank + 1)

    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_conference.iloc[idx]
        if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append(conference)

    sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

    keywords = title_input.split()
    for result in sorted_borda:
        result['Title'] = highlight_keywords(result['Title'], keywords)
        result['Description'] = highlight_keywords(result['Description'], keywords)

    num_results = len(sorted_borda)
    execution_time = time.time() - start_time

    filtered_search_results = sorted_borda

    return render_template(
        'results.html',
        results=sorted_borda,
        execution_time=execution_time,
        num_results=num_results,
        title_input=title_input,
        ranks=ranks,
        selected_field=selected_field,
        borda_scores=borda_scores,
        action=action,
        k=k
    )

@app.route('/recommend_abstract', methods=['POST'])
def recommend_abstract():
    abstract_input = request.form['abstract_input']
    ranks = request.form.get('ranks', "").split(',')
    selected_field = request.form.get('selected_field', "")
    k = int(request.form.get('num_results', 10))
    action = request.form.get('action', "recommend_abstract")

    global filtered_search_results
    filtered_search_results = []

    start_time = time.time()

    input_embedding = model.encode([abstract_input]) if abstract_input else None

    N = 100

    borda_scores = defaultdict(float)

    weight_abstract_title_vector = CRITERIA_WEIGHTS['abstract_with_title_vector']*2
    weight_abstract_description = CRITERIA_WEIGHTS['abstract_with_description_vector']*2

    if input_embedding is not None:
        labels_title_from_input, distances_title = p_title.knn_query(input_embedding, k=N)
        labels_description_from_input, distances_description = p_description.knn_query(input_embedding, k=N)

        ranking_title = {idx: rank + 1 for rank, idx in enumerate(labels_title_from_input[0])}
        ranking_description = {idx: rank + 1 for rank, idx in enumerate(labels_description_from_input[0])}

        N = N

        for idx, rank in ranking_title.items():
            borda_scores[idx] += weight_abstract_title_vector * (N - rank + 1)

        for idx, rank in ranking_description.items():
            borda_scores[idx] += weight_abstract_description * (N - rank + 1)

    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_conference.iloc[idx]
        if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append(conference)

    sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

    keywords = abstract_input.split()
    for result in sorted_borda:
        result['Title'] = highlight_keywords(result['Title'], keywords)
        result['Description'] = highlight_keywords(result['Description'], keywords)

    num_results = len(sorted_borda)
    execution_time = time.time() - start_time

    filtered_search_results = sorted_borda

    return render_template(
        'results.html',
        results=sorted_borda,
        execution_time=execution_time,
        num_results=num_results,
        abstract_input=abstract_input,
        ranks=ranks,
        selected_field=selected_field,
        borda_scores=borda_scores,
        action=action,
        k=k
    )


@app.route('/recommend_both', methods=['POST'])
def recommend_both():
    title = request.form.get('title_input')
    abstract = request.form.get('abstract_input')
    ranks = request.form.get('ranks', "").split(',')
    selected_field = request.form.get('selected_field', "")
    k = int(request.form.get('num_results', 10))
    action = request.form.get('action', "recommend_both")
    
    global filtered_search_results
    filtered_search_results = []

    start_time = time.time()

    title_embedding_input = model.encode([title])
    abstract_embedding_input = model.encode([abstract])

    N = 100

    borda_scores = defaultdict(float)

    weight_title_vector = CRITERIA_WEIGHTS['title_with_title_vector']
    weight_description_sent2vec = CRITERIA_WEIGHTS['title_with_description_vector']
    weight_abstract_title_vector = CRITERIA_WEIGHTS['abstract_with_title_vector']
    weight_abstract_description = CRITERIA_WEIGHTS['abstract_with_description_vector']

    labels_title_from_title, _ = p_title.knn_query(title_embedding_input, k=N)
    labels_description_from_title, _ = p_description.knn_query(title_embedding_input, k=N)
    labels_title_from_abstract, _ = p_title.knn_query(abstract_embedding_input, k=N)
    labels_description_from_abstract, _ = p_description.knn_query(abstract_embedding_input, k=N)

    N = N

    for rank, idx in enumerate(labels_title_from_title[0]):
        borda_scores[idx] += weight_title_vector * (N - rank)

    for rank, idx in enumerate(labels_description_from_title[0]):
        borda_scores[idx] += weight_description_sent2vec * (N - rank)

    for rank, idx in enumerate(labels_title_from_abstract[0]):
        borda_scores[idx] += weight_abstract_title_vector * (N - rank)

    for rank, idx in enumerate(labels_description_from_abstract[0]):
        borda_scores[idx] += weight_abstract_description * (N - rank)

    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_conference.iloc[idx]
        if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append(conference)

    sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

    keywords = title.split() + abstract.split()
    top_results_with_distances = []
    for result in sorted_borda:
        result_with_distances = result.copy()
        result_with_distances['Title'] = highlight_keywords(result['Title'], keywords)
        result_with_distances['Description'] = highlight_keywords(result['Description'], keywords)
        result_with_distances['Borda_Score'] = '{:.1f}'.format(borda_scores[result.name]) if result.name in borda_scores else '0.00'
        top_results_with_distances.append(result_with_distances)

    num_results = len(top_results_with_distances)
    execution_time = time.time() - start_time

    filtered_search_results = sorted_borda

    return render_template('results.html',
                            results=top_results_with_distances,
                            execution_time=execution_time,
                            num_results=num_results,
                            title_input=title,
                            abstract_input=abstract,
                            ranks=ranks,
                            selected_field=selected_field,
                            borda_scores=borda_scores,
                            action=action,
                            k=k)

@app.route('/recommend_flexible', methods=['POST'])
def recommend_flexible():
    title = request.form.get('title_input', "").strip()
    abstract = request.form.get('abstract_input', "").strip()
    ranks = request.form.get('ranks', "").split(',')
    selected_field = request.form.get('selected_field', "")
    k = int(request.form.get('num_results', 10))
    action = request.form.get('action', "recommend_flexible")
    
    global filtered_search_results
    filtered_search_results = []

    start_time = time.time()

    title_embedding_input = model.encode([title]) if title else None
    abstract_embedding_input = model.encode([abstract]) if abstract else None

    N = 100

    borda_scores = defaultdict(float)

    weight_title_vector = CRITERIA_WEIGHTS['title_with_title_vector']
    weight_description_sent2vec = CRITERIA_WEIGHTS['title_with_description_vector']
    weight_abstract_title_vector = CRITERIA_WEIGHTS['abstract_with_title_vector']
    weight_abstract_description = CRITERIA_WEIGHTS['abstract_with_description_vector']

    both_inputs_provided = bool(title and abstract)

    if title:
        labels_title_from_title, _ = p_title.knn_query(title_embedding_input, k=N)
        labels_description_from_title, _ = p_description.knn_query(title_embedding_input, k=N)
        for rank, idx in enumerate(labels_title_from_title[0]):
            borda_scores[idx] += weight_title_vector * (2 if not both_inputs_provided else 1) * (N - rank)
        for rank, idx in enumerate(labels_description_from_title[0]):
            borda_scores[idx] += weight_description_sent2vec * (2 if not both_inputs_provided else 1) * (N - rank)

    if abstract:
        labels_title_from_abstract, _ = p_title.knn_query(abstract_embedding_input, k=N)
        labels_description_from_abstract, _ = p_description.knn_query(abstract_embedding_input, k=N)
        for rank, idx in enumerate(labels_title_from_abstract[0]):
            borda_scores[idx] += weight_abstract_title_vector * (2 if not both_inputs_provided else 1) * (N - rank)
        for rank, idx in enumerate(labels_description_from_abstract[0]):
            borda_scores[idx] += weight_abstract_description * (2 if not both_inputs_provided else 1) * (N - rank)

    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_conference.iloc[idx]
        if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append(conference)

    sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

    keywords = title.split() + abstract.split()
    top_results_with_distances = []
    for result in sorted_borda:
        result_with_distances = result.copy()
        result_with_distances['Title'] = highlight_keywords(result['Title'], keywords)
        result_with_distances['Description'] = highlight_keywords(result['Description'], keywords)
        result_with_distances['Borda_Score'] = '{:.1f}'.format(borda_scores[result.name]) if result.name in borda_scores else '0.00'
        top_results_with_distances.append(result_with_distances)

    num_results = len(top_results_with_distances)
    execution_time = time.time() - start_time

    filtered_search_results = sorted_borda

    return render_template('results.html',
                            results=top_results_with_distances,
                            execution_time=execution_time,
                            num_results=num_results,
                            title_input=title,
                            abstract_input=abstract,
                            ranks=ranks,
                            selected_field=selected_field,
                            borda_scores=borda_scores,
                            action=action,
                            k=k)

@app.route('/find_conference/<int:article_id>', methods=['GET', 'POST'])
def find_conference(article_id):
    selected_article = data_article_filtered.iloc[article_id]
    title = selected_article['Title']
    abstract = selected_article['Abstract']
    k = int(request.form.get('num_results', 10))
    action = request.form.get('action', "find_conference")

    global filtered_search_results
    filtered_search_results = []

    start_time = time.time()

    title_embedding_input = model.encode([title])
    abstract_embedding_input = model.encode([abstract])

    N = 100

    borda_scores = defaultdict(float)

    weight_title_vector = CRITERIA_WEIGHTS['title_with_title_vector']
    weight_description_sent2vec = CRITERIA_WEIGHTS['title_with_description_vector']
    weight_abstract_title_vector = CRITERIA_WEIGHTS['abstract_with_title_vector']
    weight_abstract_description = CRITERIA_WEIGHTS['abstract_with_description_vector']

    labels_title_from_title, _ = p_title.knn_query(title_embedding_input, k=N)
    labels_description_from_title, _ = p_description.knn_query(title_embedding_input, k=N)
    labels_title_from_abstract, _ = p_title.knn_query(abstract_embedding_input, k=N)
    labels_description_from_abstract, _ = p_description.knn_query(abstract_embedding_input, k=N)

    N = N

    for rank, idx in enumerate(labels_title_from_title[0]):
        borda_scores[idx] += weight_title_vector * (N - rank)

    for rank, idx in enumerate(labels_description_from_title[0]):
        borda_scores[idx] += weight_description_sent2vec * (N - rank)

    for rank, idx in enumerate(labels_title_from_abstract[0]):
        borda_scores[idx] += weight_abstract_title_vector * (N - rank)

    for rank, idx in enumerate(labels_description_from_abstract[0]):
        borda_scores[idx] += weight_abstract_description * (N - rank)

    sorted_borda = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = [data_conference.iloc[idx] for idx, _ in sorted_borda[:k]]

    keywords = title.split() + abstract.split()
    top_results_with_distances = []
    for result in top_results:
        result_with_distances = result.copy()
        result_with_distances['Title'] = highlight_keywords(result['Title'], keywords)
        result_with_distances['Description'] = highlight_keywords(result['Description'], keywords)
        result_with_distances['Borda_Score'] = '{:.1f}'.format(borda_scores[result.name]) if result.name in borda_scores else '0.00'
        top_results_with_distances.append(result_with_distances)

    num_results = len(top_results_with_distances)

    filtered_search_results = top_results_with_distances
    execution_time = time.time() - start_time

    return render_template('results.html', results=top_results_with_distances,
                        execution_time=execution_time, num_results=num_results,
                        title_input=title, abstract_input=abstract,
                        borda_scores=borda_scores, action=action,
                        k=k, article_id=article_id)


@app.route('/filter', methods=['POST'])
def filter():
    title_input = request.form.get('title_input', "")
    abstract_input = request.form.get('abstract_input', "")
    ranks = request.form.getlist('ranks')
    selected_field = request.form.get('selected_field', None)
    k = int(request.form.get('num_results', 10))
    action = request.form.get('action', "filter")

    global filtered_search_results
    filtered_search_results = []

    N = 100

    borda_scores = defaultdict(float)

    if title_input and abstract_input:
        rank_weights = {
            'title': CRITERIA_WEIGHTS['title_with_title_vector'],
            'description': CRITERIA_WEIGHTS['title_with_description_vector'],
            'abstract_title': CRITERIA_WEIGHTS['abstract_with_title_vector'],
            'abstract_description': CRITERIA_WEIGHTS['abstract_with_description_vector']
        }
    elif title_input:
        rank_weights = {
            'title': CRITERIA_WEIGHTS['title_with_title_vector']*2,
            'description': CRITERIA_WEIGHTS['title_with_description_vector']*2
        }
    elif abstract_input:
        rank_weights = {
            'abstract_title': CRITERIA_WEIGHTS['abstract_with_title_vector']*2,
            'abstract_description': CRITERIA_WEIGHTS['abstract_with_description_vector']*2
        }
    else:
        return render_template('results.html', results=[], execution_time=0, num_results=0)

    start_time = time.time()

    title_embedding_input = model.encode([title_input]) if title_input else None
    abstract_embedding_input = model.encode([abstract_input]) if abstract_input else None

    if title_embedding_input is not None:
        labels_title, _ = p_title.knn_query(title_embedding_input, k=N)
        labels_description, _ = p_description.knn_query(title_embedding_input, k=N)
        for i, idx in enumerate(labels_title[0]):
            borda_scores[idx] += rank_weights.get('title', 0) * (N - i)
        for i, idx in enumerate(labels_description[0]):
            borda_scores[idx] += rank_weights.get('description', 0) * (N - i)

    if abstract_embedding_input is not None:
        labels_title_abstract, _ = p_title.knn_query(abstract_embedding_input, k=N)
        labels_description_abstract, _ = p_description.knn_query(abstract_embedding_input, k=N)
        for i, idx in enumerate(labels_title_abstract[0]):
            borda_scores[idx] += rank_weights.get('abstract_title', 0) * (N - i)
        for i, idx in enumerate(labels_description_abstract[0]):
            borda_scores[idx] += rank_weights.get('abstract_description', 0) * (N - i)

    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_conference.iloc[idx]
        if (not ranks or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append((conference, score))

    sorted_borda = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:k]

    keywords = title_input.split() + abstract_input.split()
    top_results_with_distances = []
    for conference, score in sorted_borda:
        result_with_distances = conference.copy()
        result_with_distances['Title'] = highlight_keywords(conference['Title'], keywords)
        result_with_distances['Description'] = highlight_keywords(conference['Description'], keywords)
        result_with_distances['Borda_Score'] = '{:.1f}'.format(score)
        top_results_with_distances.append(result_with_distances)

    filtered_search_results = top_results_with_distances
    execution_time = time.time() - start_time

    return render_template(
        'results.html',
        results=top_results_with_distances,
        execution_time=execution_time,
        num_results=len(top_results_with_distances),
        title_input=title_input,
        abstract_input=abstract_input,
        ranks=ranks,
        selected_field=selected_field,
        borda_scores=borda_scores,
        action=action,
        k=k
    )


@app.route('/conference/<int:article_id>')
def conference_detail(article_id):
    conference = get_conference_by_id(article_id)

    if conference:
        return render_template('conference.html', conference=conference)
    else:
        return render_template('404.html'), 404

@app.route('/export_csv', methods=['GET'])
def export_csv():
    export_all = request.args.get('export_all', 'false').lower() == 'true'

    if export_all:
        try:
            df = pd.read_csv('data_conference.csv')
            columns_to_exclude = ['Title_Vector', 'Description_Sent2vec', 'Conference_Link', 'Conference_Category', 'Borda_Score']
            df = df.drop(columns=columns_to_exclude, errors='ignore')
        except FileNotFoundError:
            return "File data_conference.csv not found", 404
    else:
        global filtered_search_results
        if not filtered_search_results:
            return "No results to export", 404

        df = pd.DataFrame(filtered_search_results)

        columns_to_exclude = ['Title_Vector', 'Description_Sent2vec', 'Conference_Link', 'Conference_Category', 'Borda_Score']
        df = df.drop(columns=columns_to_exclude, errors='ignore')

        df['Title'] = df['Title'].apply(lambda x: clean_html(x))
        df['Description'] = df['Description'].apply(lambda x: clean_html(x))

    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=conference.csv"
    response.headers["Content-Type"] = "text/csv"

    return response

def clean_html(text):
    """Remove HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


import bcrypt
app.secret_key = os.urandom(24)

@app.route('/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')

        with open('login.txt', 'r') as f:
            credentials = f.readlines()

        for credential in credentials:
            stored_username, stored_hashed_password = credential.strip().split(':')
            if username == stored_username:
                if bcrypt.checkpw(password, stored_hashed_password.encode('utf-8')):
                    session['logged_in'] = True
                    return redirect(url_for('admin'))

        flash('Incorrect username or password!', 'danger')

    return render_template('login.html')

import os
import pandas as pd
from datetime import datetime
from flask import render_template, flash

@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        flash('You need to login to access this page!', 'danger')
        return redirect(url_for('login'))

    try:
        file_path = 'data_conference.csv'
        data = pd.read_csv(file_path)

        columns = data.columns.tolist()

        conferences = data.to_dict(orient='records')

        num_rows = data.shape[0]
        num_columns = data.shape[1]

        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        file_mod_time = os.path.getmtime(file_path)
        last_modified_date = datetime.fromtimestamp(file_mod_time).strftime('%d-%m-%Y %H:%M:%S')

    except Exception as e:
        flash(f'Error reading CSV file: {e}', 'danger')
        columns = []
        conferences = []
        num_rows = num_columns = 0
        file_size_mb = last_modified_date = ''

    return render_template(
        'admin.html',
        conferences=conferences,
        columns=columns,
        num_rows=num_rows,
        num_columns=num_columns,
        file_size_mb=file_size_mb,
        last_modified_date=last_modified_date
    )

@app.route('/admin_article')
def admin_article():
    try:
        file_path = 'data_article.csv'
        data = pd.read_csv(file_path)

        columns = data.columns.tolist()

        articles = data.to_dict(orient='records')

        num_rows = data.shape[0]
        num_columns = data.shape[1]

        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        file_mod_time = os.path.getmtime(file_path)
        last_modified_date = datetime.fromtimestamp(file_mod_time).strftime('%d-%m-%Y %H:%M:%S')

    except Exception as e:
        flash(f'Error reading Excel file: {e}', 'danger')
        columns = []
        articles = []
        num_rows = num_columns = 0
        file_size_mb = last_modified_date = ''

    return render_template(
        'admin_article.html',
        articles=articles,
        columns=columns,
        num_rows=num_rows,
        num_columns=num_columns,
        file_size_mb=file_size_mb,
        last_modified_date=last_modified_date
    )

UPLOAD_FOLDER = os.path.dirname(__file__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

from tqdm import tqdm

def process_and_vectorize_with_progress(texts):
    vectors = []
    for text in tqdm(texts, desc="Processing and Vectorizing"):
        preprocessed_text = preprocess_text(str(text))
        sentences = sent_tokenize(preprocessed_text)
        sentence_vectors = np.array([model.encode(sentence) for sentence in sentences])
        vectors.append(sentence_vectors.tolist())
    return vectors

@app.route('/import_conference', methods=['POST'])
def upload_conference_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('admin'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('admin'))

    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f'Error reading file: {e}', 'error')
            return redirect(url_for('admin'))

        required_columns = ['ERAID', 'Title', 'Description', 'Rank', 'Acronym', 'FoR1 Name', 'Conference_Link']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            flash(f"File is missing required columns: {', '.join(missing_columns)}", 'error')
            return redirect(url_for('admin'))

        if len(df) < 99:
            flash('File must contain more than 100 rows', 'error')
            return redirect(url_for('admin'))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data_conference.csv')
        file.save(file_path)

        if 'Title' in df.columns and 'Title_Vector' not in df.columns:
            df['Title_Vector'] = process_and_vectorize_with_progress(df['Title'])

        if 'Description' in df.columns and 'Description_Sent2vec' not in df.columns:
            df['Description_Sent2vec'] = process_and_vectorize_with_progress(df['Description'])

        df.to_csv(file_path, index=False)
        reload_system()

        flash('File uploaded and imported successfully', 'success')
        return redirect(url_for('admin'))

    flash('Invalid file format', 'error')
    return redirect(url_for('admin'))

def reload_system():
    global model, data_conference, p_title, p_description
    print("Reloading model and data...")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    data_conference = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'data_conference.csv'))
    data_conference.fillna('', inplace=True)

    data_conference['Title_Vector'] = data_conference['Title_Vector'].apply(safe_json_to_array)
    data_conference['Description_Sent2vec'] = data_conference['Description_Sent2vec'].apply(safe_json_to_array)

    valid_title_embeddings = data_conference['Title_Vector'].apply(lambda x: len(x) > 0)
    valid_description_embeddings = data_conference['Description_Sent2vec'].apply(lambda x: len(x) > 0)

    data_title_filtered = data_conference[valid_title_embeddings].reset_index(drop=True)
    data_description_filtered = data_conference[valid_description_embeddings].reset_index(drop=True)

    title_embeddings = np.vstack(data_title_filtered['Title_Vector'].values)
    description_embeddings = np.vstack(data_description_filtered['Description_Sent2vec'].values)

    p_title = hnswlib.Index(space='cosine', dim=title_embeddings.shape[1])
    p_title.init_index(max_elements=len(title_embeddings), ef_construction=300, M=20)
    p_title.add_items(title_embeddings)
    p_title.set_ef(50)

    p_description = hnswlib.Index(space='cosine', dim=description_embeddings.shape[1])
    p_description.init_index(max_elements=len(description_embeddings), ef_construction=300, M=20)
    p_description.add_items(description_embeddings)
    p_description.set_ef(50)

    print("System reload complete.")

@app.route('/export_conference', methods=['GET'])
def export_conference():
    try:
        df = pd.read_csv('data_conference.csv')
    except FileNotFoundError:
        return "File data_conference.csv not found", 404

    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=conference_data.csv"
    response.headers["Content-Type"] = "text/csv"

    return response

UPLOAD_FOLDER = os.path.dirname(__file__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def process_and_vectorize(text):
    preprocessed_text = preprocess_text(str(text))
    sentences = sent_tokenize(preprocessed_text)
    sentence_vectors = np.array([model.encode(sentence) for sentence in sentences])
    return sentence_vectors.tolist()

@app.route('/import_article', methods=['POST'])
def import_article_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('admin_article'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('admin_article'))

    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f'Error reading the file: {str(e)}', 'error')
            return redirect(url_for('admin_article'))

        required_columns = {'STT', 'Title', 'Abstract'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            flash(f'File is missing required columns: {", ".join(missing_columns)}', 'error')
            return redirect(url_for('admin_article'))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data_article.csv')
        file.save(file_path)
        
        flash('File uploaded successfully!', 'success')

        if 'Title' in df.columns and 'Title_Vector' not in df.columns:
            df['Title_Vector'] = df['Title'].apply(process_and_vectorize)

        if 'Abstract' in df.columns and 'Abstract_Sent2vec' not in df.columns:
            df['Abstract_Sent2vec'] = df['Abstract'].apply(process_and_vectorize)

        df.to_csv(file_path, index=False)

        reload_system2()
        index(page=1)

        print("Vector hóa thành công!")
    else:
        flash('Only CSV files are allowed', 'error')

    return redirect(url_for('admin_article'))

def reload_system2():
    global model, data_articles, p_title, p_abstract
    print("Reloading model and data...")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    data_articles = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'data_article.csv'))
    data_articles.fillna('', inplace=True)

    data_articles['Title_Vector'] = data_articles['Title_Vector'].apply(safe_json_to_array)
    data_articles['Abstract_Sent2vec'] = data_articles['Abstract_Sent2vec'].apply(safe_json_to_array)

    valid_article_title_embeddings = data_articles['Title_Vector'].apply(lambda x: len(x) > 0)
    valid_article_abstract_embeddings = data_articles['Abstract_Sent2vec'].apply(lambda x: len(x) > 0)

    data_article_filtered = data_articles[valid_article_title_embeddings].reset_index(drop=True)

    article_title_embeddings = np.vstack(data_article_filtered['Title_Vector'].values)
    article_abstract_embeddings = np.vstack(data_article_filtered['Abstract_Sent2vec'].values)

    dimension_article_title = article_title_embeddings.shape[1]
    p_article_title = hnswlib.Index(space='cosine', dim=dimension_article_title)
    p_article_title.init_index(max_elements=len(article_title_embeddings), ef_construction=200, M=20)
    p_article_title.add_items(article_title_embeddings)
    p_article_title.set_ef(50)

    dimension_article_abstract = article_abstract_embeddings.shape[1]
    p_article_abstract = hnswlib.Index(space='cosine', dim=dimension_article_abstract)
    p_article_abstract.init_index(max_elements=len(article_abstract_embeddings), ef_construction=200, M=20)
    p_article_abstract.add_items(article_abstract_embeddings)
    p_article_abstract.set_ef(50)

    print("System reload complete.")

@app.route('/admin_setting')
def setting():
    return render_template('admin_setting.html', weights=CRITERIA_WEIGHTS)

@app.route('/export_article', methods=['GET'])
def export_article():
    try:
        df = pd.read_csv('data_article.csv')
    except FileNotFoundError:
        return "File not found", 404

    csv_data = df.to_csv(index=False, encoding='utf-8-sig')

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=data_article.csv"
    response.headers["Content-Type"] = "text/csv"

    return response

@app.route('/view_conference')
def view_conference():
    conference_data = pd.read_csv('data_conference.csv')
    conference_records = conference_data.to_dict(orient='records')
    return render_template('admin.html', conferences=conference_records)


@app.route('/logout')
def logout():
    session.clear()
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/legal')
def legal():
    return render_template('legal.html')

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
