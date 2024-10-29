from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import hnswlib
from collections import defaultdict
import time

app = Flask(__name__)

# Load data and model
model = SentenceTransformer('all-MiniLM-L6-v2')
data_reference = pd.read_csv('Data.csv')  # Dữ liệu hội thảo
data_articles = pd.read_excel('data-test.xlsx')  # Dữ liệu bài báo

# Thay thế toàn bộ giá trị NaN bằng chuỗi rỗng
data_reference.fillna('', inplace=True)

# Preprocess conference data 'Data-2.xlsx'
def safe_json_to_array(x):
    if isinstance(x, str):  # Kiểm tra xem x có phải là chuỗi không
        try:
            return np.array(json.loads(x))
        except json.JSONDecodeError as e:
            print(f"Invalid string: {x}. Error: {e}")
            return np.array([])
    else:
        return np.array([])

data_reference['Title_Vector'] = data_reference['Title_Vector'].apply(safe_json_to_array)
data_reference['Description_Sent2vec'] = data_reference['Description_Sent2vec'].apply(safe_json_to_array)

valid_title_embeddings = data_reference['Title_Vector'].apply(lambda x: len(x) > 0)
valid_description_embeddings = data_reference['Description_Sent2vec'].apply(lambda x: len(x) > 0)

data_title_filtered = data_reference[valid_title_embeddings].reset_index(drop=True)
data_description_filtered = data_reference[valid_description_embeddings].reset_index(drop=True)

title_embeddings = np.vstack(data_title_filtered['Title_Vector'].values)
description_embeddings = np.vstack(data_description_filtered['Description_Sent2vec'].values)

# Initialize HNSW for conferences
dimension_title = title_embeddings.shape[1]
p_title = hnswlib.Index(space='cosine', dim=dimension_title)
p_title.init_index(max_elements=len(title_embeddings), ef_construction=200, M=16)
p_title.add_items(title_embeddings)
p_title.set_ef(50)

dimension_description = description_embeddings.shape[1]
p_description = hnswlib.Index(space='cosine', dim=dimension_description)
p_description.init_index(max_elements=len(description_embeddings), ef_construction=200, M=16)
p_description.add_items(description_embeddings)
p_description.set_ef(50)

# Preprocess article data 'data-test.xlsx'
data_articles['Title_Vector'] = data_articles['Title_Vector'].apply(safe_json_to_array)
data_articles['Abstract_Sent2vec'] = data_articles['Abstract_Sent2vec'].apply(safe_json_to_array)

valid_article_title_embeddings = data_articles['Title_Vector'].apply(lambda x: len(x) > 0)
valid_article_abstract_embeddings = data_articles['Abstract_Sent2vec'].apply(lambda x: len(x) > 0)

data_article_filtered = data_articles[valid_article_title_embeddings].reset_index(drop=True)

article_title_embeddings = np.vstack(data_article_filtered['Title_Vector'].values)
article_abstract_embeddings = np.vstack(data_article_filtered['Abstract_Sent2vec'].values)

# Initialize HNSW for articles
dimension_article_title = article_title_embeddings.shape[1]
p_article_title = hnswlib.Index(space='cosine', dim=dimension_article_title)
p_article_title.init_index(max_elements=len(article_title_embeddings), ef_construction=200, M=16)
p_article_title.add_items(article_title_embeddings)
p_article_title.set_ef(50)

dimension_article_abstract = article_abstract_embeddings.shape[1]
p_article_abstract = hnswlib.Index(space='cosine', dim=dimension_article_abstract)
p_article_abstract.init_index(max_elements=len(article_abstract_embeddings), ef_construction=200, M=16)
p_article_abstract.add_items(article_abstract_embeddings)
p_article_abstract.set_ef(50)

# Hàm để lấy thông tin hội thảo dựa trên article_id
def get_conference_by_id(article_id):
    # Giả sử article_id tương ứng với cột 'ERAID' trong file 'Data-2.xlsx'
    conference_row = data_reference[data_reference['ERAID'] == article_id]

    if not conference_row.empty:
        # Nếu tìm thấy, trả về dữ liệu hàng đầu tiên dưới dạng dictionary
        return conference_row.iloc[0].to_dict()
    else:
        # Nếu không tìm thấy, trả về None
        return None

@app.route('/')
def index():
    articles = data_article_filtered.to_dict(orient='records')
    return render_template('index.html', articles=articles, title_input='', abstract_input='')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     title_input = request.form.get('title_input', "")
#     ranks = request.form.get('ranks', "").split(',')
#     selected_field = request.form.get('selected_field', "")

#     start_time = time.time()

#     # Mã hóa title_input
#     input_embedding = model.encode([title_input]) if title_input else None

#     k = 10  # số lượng kết quả hiển thị

#     borda_scores = defaultdict(float)

#     # Tạo trọng số cho từng tiêu chí (MCDM)
#     weight_title_vector = 0.7  # Trọng số cho tiêu đề
#     weight_description_sent2vec = 0.4  # Trọng số cho mô tả

#     if input_embedding is not None:
#         # Tìm kiếm hội thảo dựa trên tiêu đề
#         labels_title_from_input, _ = p_title.knn_query(input_embedding, k=k)
#         labels_description_from_input, _ = p_description.knn_query(input_embedding, k=k)

#         # Borda Count cho tiêu đề
#         for i in range(len(labels_title_from_input[0])):
#             idx = labels_title_from_input[0][i]
#             borda_scores[idx] += (weight_title_vector * (k - i))

#         # Borda Count cho mô tả
#         for i in range(len(labels_description_from_input[0])):
#             idx = labels_description_from_input[0][i]
#             borda_scores[idx] += (weight_description_sent2vec * (k - i))

#     # Lọc kết quả nếu có giá trị cho ranks hoặc selected_field
#     filtered_results = []
#     for idx, score in borda_scores.items():
#         conference = data_reference.iloc[idx]
#         if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
#             filtered_results.append(conference)

#     # Sắp xếp và lấy top k kết quả sau khi lọc
#     sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

#     num_results = len(sorted_borda)
#     execution_time = time.time() - start_time

#     return render_template(
#         'results.html',
#         results=sorted_borda,
#         execution_time=execution_time,
#         num_results=num_results,
#         title_input=title_input,
#         borda_scores=borda_scores
#     )

@app.route('/recommend', methods=['POST'])
def recommend():
    title_input = request.form.get('title_input', "")
    ranks = request.form.get('ranks', "").split(',')
    selected_field = request.form.get('selected_field', "")

    start_time = time.time()

    # Mã hóa title_input
    input_embedding = model.encode([title_input]) if title_input else None

    k = 10  # Số lượng kết quả hiển thị cuối cùng

    initial_k = min(len(data_reference), k * 5)  # Tăng số lượng để đảm bảo đủ kết quả

    borda_scores = defaultdict(float)

    # Tạo trọng số cho từng tiêu chí (MCDM)
    weight_title_vector = 0.7  # Trọng số cho tiêu đề
    weight_description_sent2vec = 0.4  # Trọng số cho mô tả

    if input_embedding is not None:
        # Tìm kiếm hội thảo dựa trên tiêu đề với initial_k kết quả ban đầu
        labels_title_from_input, _ = p_title.knn_query(input_embedding, k=initial_k)
        labels_description_from_input, _ = p_description.knn_query(input_embedding, k=initial_k)

        # Borda Count cho tiêu đề
        for i in range(len(labels_title_from_input[0])):
            idx = labels_title_from_input[0][i]
            borda_scores[idx] += (weight_title_vector * (initial_k - i))

        # Borda Count cho mô tả
        for i in range(len(labels_description_from_input[0])):
            idx = labels_description_from_input[0][i]
            borda_scores[idx] += (weight_description_sent2vec * (initial_k - i))

    # Lọc kết quả nếu có giá trị cho ranks hoặc selected_field
    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_reference.iloc[idx]
        if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append(conference)

    # Sắp xếp và lấy top k kết quả sau khi lọc
    sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

    num_results = len(sorted_borda)
    execution_time = time.time() - start_time

    return render_template(
        'results.html',
        results=sorted_borda,
        execution_time=execution_time,
        num_results=num_results,
        title_input=title_input,
        ranks = ranks,
        selected_field = selected_field,
        borda_scores=borda_scores
    )

@app.route('/recommend_abstract', methods=['POST'])
def recommend_abstract():
    abstract_input= request.form['abstract_input']  # Lấy dữ liệu từ ô nhập liệu cho abstract
    ranks = request.form.get('ranks', "").split(',')
    selected_field = request.form.get('selected_field', "")

    start_time = time.time()

    # Mã hóa đầu vào abstract
    input_embedding = model.encode([abstract_input]) if abstract_input else None

    k = 10  # Số lượng kết quả hiển thị cuối cùng

    initial_k = min(len(data_reference), k * 5)  # Tăng số lượng để đảm bảo đủ kết quả

    borda_scores = defaultdict(float)

    # Tạo trọng số cho từng tiêu chí (MCDM)
    weight_title_vector = 0.7  # Trọng số cho tiêu đề
    weight_description_sent2vec = 0.4  # Trọng số cho mô tả

    if input_embedding is not None:
        # Tìm kiếm hội thảo dựa trên tiêu đề với initial_k kết quả ban đầu
        labels_title_from_input, _ = p_title.knn_query(input_embedding, k=initial_k)
        labels_description_from_input, _ = p_description.knn_query(input_embedding, k=initial_k)

        # Borda Count cho tiêu đề
        for i in range(len(labels_title_from_input[0])):
            idx = labels_title_from_input[0][i]
            borda_scores[idx] += (weight_title_vector * (initial_k - i))

        # Borda Count cho mô tả
        for i in range(len(labels_description_from_input[0])):
            idx = labels_description_from_input[0][i]
            borda_scores[idx] += (weight_description_sent2vec * (initial_k - i))

    # Lọc kết quả nếu có giá trị cho ranks hoặc selected_field
    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_reference.iloc[idx]
        if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append(conference)

    # Sắp xếp và lấy top k kết quả sau khi lọc
    sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

    num_results = len(sorted_borda)
    execution_time = time.time() - start_time

    return render_template(
        'results.html',
        results=sorted_borda,
        execution_time=execution_time,
        num_results=num_results,
        abstract_input=abstract_input,
        ranks = ranks,
        selected_field = selected_field,
        borda_scores=borda_scores
    )

@app.route('/recommend_both', methods=['POST'])
def recommend_both():
    title = request.form.get('title_input')
    abstract = request.form.get('abstract_input')
    ranks = request.form.get('ranks', "").split(',')
    selected_field = request.form.get('selected_field', "")

    start_time = time.time()

    # Mã hóa tiêu đề và tóm tắt
    title_embedding_input = model.encode([title])
    abstract_embedding_input = model.encode([abstract])

    k = 10  # Số lượng kết quả hiển thị
    initial_k = min(len(data_reference), k * 5)

    borda_scores = defaultdict(float)

    # Tạo trọng số cho từng tiêu chí (MCDM)
    weight_title_vector = 0.7
    weight_description_sent2vec = 0.4
    weight_abstract_title_vector = 0.5
    weight_abstract_description = 0.7

    # Tìm kiếm và Borda Count cho các tiêu chí
    labels_title_from_title, _ = p_title.knn_query(title_embedding_input, k=initial_k)
    labels_description_from_title, _ = p_description.knn_query(title_embedding_input, k=initial_k)
    labels_title_from_abstract, _ = p_title.knn_query(abstract_embedding_input, k=initial_k)
    labels_description_from_abstract, _ = p_description.knn_query(abstract_embedding_input, k=initial_k)

    for i in range(len(labels_title_from_title[0])):
        idx = labels_title_from_title[0][i]
        borda_scores[idx] += (initial_k - i) * weight_title_vector

    for i in range(len(labels_description_from_title[0])):
        idx = labels_description_from_title[0][i]
        borda_scores[idx] += (initial_k - i) * weight_description_sent2vec

    for i in range(len(labels_title_from_abstract[0])):
        idx = labels_title_from_abstract[0][i]
        borda_scores[idx] += (initial_k - i) * weight_abstract_title_vector

    for i in range(len(labels_description_from_abstract[0])):
        idx = labels_description_from_abstract[0][i]
        borda_scores[idx] += (initial_k - i) * weight_abstract_description

    # Lọc và sắp xếp kết quả theo ranks và selected_field
    filtered_results = []
    for idx, score in borda_scores.items():
        conference = data_reference.iloc[idx]
        if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
            filtered_results.append(conference)

    sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

    # Chuẩn bị kết quả với điểm Borda
    top_results_with_distances = []
    for idx, result in enumerate(sorted_borda):
        result_with_distances = result.copy()
        result_with_distances['Borda_Score'] = '{:.1f}'.format(borda_scores[result.name]) if result.name in borda_scores else '0.00'
        top_results_with_distances.append(result_with_distances)

    num_results = len(top_results_with_distances)  # Số lượng kết quả tìm được
    execution_time = time.time() - start_time

    # Truyền dữ liệu vào template
    return render_template('results.html',
                            results=top_results_with_distances,
                            execution_time=execution_time,
                            num_results=num_results,
                            title_input=title,
                            abstract_input=abstract,
                            ranks=ranks,
                            selected_field=selected_field,
                            borda_scores=borda_scores)

# @app.route('/recommend_both', methods=['POST'])
# def recommend_both():
#     title = request.form.get('title_input')
#     abstract = request.form.get('abstract_input')
#     ranks = request.form.get('ranks', "").split(',')
#     selected_field = request.form.get('selected_field', "")

#     start_time = time.time()

#     # Mã hóa tiêu đề và tóm tắt
#     title_embedding_input = model.encode([title]) if title else None
#     abstract_embedding_input = model.encode([abstract]) if abstract else None

#     k = 10  # Số lượng kết quả hiển thị
#     initial_k = min(len(data_reference), k * 5)

#     borda_scores = defaultdict(float)

#     # Tạo trọng số cho từng tiêu chí (MCDM)
#     weight_title_vector = 0.7
#     weight_description_sent2vec = 0.4
#     weight_abstract_title_vector = 0.5
#     weight_abstract_description = 0.7

#     if title_embedding_input is not None:
#         # Tìm kiếm dựa trên tiêu đề
#         labels_title_from_title, _ = p_title.knn_query(title_embedding_input, k=initial_k)
#         labels_description_from_title, _ = p_description.knn_query(title_embedding_input, k=initial_k)

#         # Borda Count cho tiêu đề và mô tả từ tiêu đề
#         for i in range(len(labels_title_from_title[0])):
#             idx = labels_title_from_title[0][i]
#             borda_scores[idx] += (initial_k - i) * weight_title_vector

#         for i in range(len(labels_description_from_title[0])):
#             idx = labels_description_from_title[0][i]
#             borda_scores[idx] += (initial_k - i) * weight_description_sent2vec

#     if abstract_embedding_input is not None:
#         # Tìm kiếm dựa trên tóm tắt
#         labels_title_from_abstract, _ = p_title.knn_query(abstract_embedding_input, k=initial_k)
#         labels_description_from_abstract, _ = p_description.knn_query(abstract_embedding_input, k=initial_k)

#         # Borda Count cho tiêu đề và mô tả từ tóm tắt
#         for i in range(len(labels_title_from_abstract[0])):
#             idx = labels_title_from_abstract[0][i]
#             borda_scores[idx] += (initial_k - i) * weight_abstract_title_vector

#         for i in range(len(labels_description_from_abstract[0])):
#             idx = labels_description_from_abstract[0][i]
#             borda_scores[idx] += (initial_k - i) * weight_abstract_description

#     # Lọc kết quả nếu có giá trị cho ranks hoặc selected_field
#     filtered_results = []
#     for idx, score in borda_scores.items():
#         conference = data_reference.iloc[idx]
#         if (not ranks[0] or conference['Rank'] in ranks) and (not selected_field or conference['FoR1 Name'] == selected_field):
#             filtered_results.append(conference)

#     # Sắp xếp và lấy top k kết quả sau khi lọc
#     sorted_borda = sorted(filtered_results, key=lambda x: borda_scores[x.name], reverse=True)[:k]

#     # Chuẩn bị kết quả với điểm Borda và thông tin khác
#     top_results_with_distances = []
#     for idx, result in enumerate(sorted_borda):
#         result_with_distances = result.copy()
#         result_with_distances['Borda_Score'] = '{:.1f}'.format(borda_scores[result.name]) if result.name in borda_scores else '0.00'
#         top_results_with_distances.append(result_with_distances)

#     num_results = len(top_results_with_distances)  # Số lượng kết quả tìm được
#     execution_time = time.time() - start_time

#     # Truyền thêm 'title' và 'abstract' vào template
#     return render_template('results.html',
#                             results=top_results_with_distances,
#                             execution_time=execution_time,
#                             num_results=num_results,
#                             title_input=title,
#                             abstract_input=abstract,
#                             ranks=ranks,
#                             selected_field=selected_field,
#                             borda_scores=borda_scores)

@app.route('/find_conference/<int:article_id>')
def find_conference(article_id):
    # Lấy tiêu đề và tóm tắt từ bài báo đã chọn
    selected_article = data_article_filtered.iloc[article_id]
    title = selected_article['Title']
    abstract = selected_article['Abstract']

    start_time = time.time()

    # Mã hóa tiêu đề và tóm tắt
    title_embedding_input = model.encode([title])
    abstract_embedding_input = model.encode([abstract])

    k = 10  # Số lượng kết quả hiển thị
    initial_k = min(len(data_reference), k * 5)

    # Tìm labels và distances cho tiêu đề và tóm tắt
    labels_title_from_title, distances_title_from_title = p_title.knn_query(title_embedding_input, k=initial_k)
    labels_description_from_title, distances_description_from_title = p_description.knn_query(title_embedding_input, k=initial_k)
    labels_title_from_abstract, distances_title_from_abstract = p_title.knn_query(abstract_embedding_input, k=initial_k)
    labels_description_from_abstract, distances_description_from_abstract = p_description.knn_query(abstract_embedding_input, k=initial_k)

    # Tạo trọng số cho từng tiêu chí (MCDM)
    weight_title_vector = 0.7
    weight_description_sent2vec = 0.4
    weight_abstract_title_vector = 0.5
    weight_abstract_description = 0.7

    borda_scores = defaultdict(float)

    # Borda Count cho tiêu đề (sử dụng trọng số)
    for i in range(len(labels_title_from_title[0])):
        idx = labels_title_from_title[0][i]
        borda_scores[idx] += (initial_k - i) * weight_title_vector

    # Borda Count cho mô tả từ tiêu đề (sử dụng trọng số)
    for i in range(len(labels_description_from_title[0])):
        idx = labels_description_from_title[0][i]
        borda_scores[idx] += (initial_k - i) * weight_description_sent2vec

    # Borda Count cho tiêu đề từ tóm tắt (sử dụng trọng số)
    for i in range(len(labels_title_from_abstract[0])):
        idx = labels_title_from_abstract[0][i]
        borda_scores[idx] += (initial_k - i) * weight_abstract_title_vector

    # Borda Count cho mô tả từ tóm tắt (sử dụng trọng số)
    for i in range(len(labels_description_from_abstract[0])):
        idx = labels_description_from_abstract[0][i]
        borda_scores[idx] += (initial_k - i) * weight_abstract_description

    # Sắp xếp Borda Score
    sorted_borda = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = [data_reference.iloc[idx] for idx, _ in sorted_borda[:k]]

    # Thêm khoảng cách vào kết quả để hiển thị
    top_results_with_distances = []
    for idx, result in enumerate(top_results):
        result_with_distances = result.copy()
        result_with_distances['Distance_Title_Title'] = distances_title_from_title[0][idx]
        result_with_distances['Distance_Desc_Title'] = distances_description_from_title[0][idx]
        result_with_distances['Distance_Title_Abstract'] = distances_title_from_abstract[0][idx]
        result_with_distances['Distance_Desc_Abstract'] = distances_description_from_abstract[0][idx]
        result_with_distances['Borda_Score'] = '{:.1f}'.format(borda_scores[result.name]) if result.name in borda_scores else '0.00'  # Thêm điểm Borda
        top_results_with_distances.append(result_with_distances)

    num_results = len(top_results_with_distances)  # Số lượng kết quả tìm được
    execution_time = time.time() - start_time

    # Truyền thêm 'title' vào template
    return render_template('results.html', results=top_results_with_distances,
                            execution_time=execution_time, num_results=num_results,
                            title_input=title, abstract_input=abstract,
                            borda_scores=borda_scores)

@app.route('/filter', methods=['POST'])
def filter_results():
    ranks = request.form.getlist('ranks')  # Lấy danh sách các giá trị Rank từ form
    selected_field = request.form.get('field', None)
    title_input = request.form['title_input']
    abstract_input = request.form['abstract_input']

    k = 10
    initial_k = min(len(data_reference), k * 5)

    borda_scores = defaultdict(float)

    # Nếu có đủ cả tiêu đề và tóm tắt
    if title_input and abstract_input:
        # Mã hóa tiêu đề và tóm tắt
        title_embedding_input = model.encode([title_input])
        abstract_embedding_input = model.encode([abstract_input])

        # Tìm labels và distances cho tiêu đề và tóm tắt
        labels_title_from_title, distances_title_from_title = p_title.knn_query(title_embedding_input, k=initial_k)
        labels_description_from_title, distances_description_from_title = p_description.knn_query(title_embedding_input, k=initial_k)
        labels_title_from_abstract, distances_title_from_abstract = p_title.knn_query(abstract_embedding_input, k=initial_k)
        labels_description_from_abstract, distances_description_from_abstract = p_description.knn_query(abstract_embedding_input, k=initial_k)

        # Borda Count cho tiêu đề từ tiêu đề
        for i in range(len(labels_title_from_title[0])):
            idx = labels_title_from_title[0][i]
            borda_scores[idx] += (0.7 * (initial_k - i))  # Trọng số cho tiêu đề từ tiêu đề

        # Borda Count cho mô tả từ tiêu đề
        for i in range(len(labels_description_from_title[0])):
            idx = labels_description_from_title[0][i]
            borda_scores[idx] += (0.4 * (initial_k - i))  # Trọng số cho mô tả từ tiêu đề

        # Borda Count cho tiêu đề từ tóm tắt
        for i in range(len(labels_title_from_abstract[0])):
            idx = labels_title_from_abstract[0][i]
            borda_scores[idx] += (0.5 * (initial_k - i))  # Trọng số cho tiêu đề từ tóm tắt

        # Borda Count cho mô tả từ tóm tắt
        for i in range(len(labels_description_from_abstract[0])):
            idx = labels_description_from_abstract[0][i]
            borda_scores[idx] += (0.7 * (initial_k - i))  # Trọng số cho mô tả từ tóm tắt

    # Nếu chỉ có tiêu đề
    elif title_input:
        title_embedding_input = model.encode([title_input])
        # Tìm labels và distances cho tiêu đề
        labels_title_from_title, distances_title_from_title = p_title.knn_query(title_embedding_input, k=initial_k)
        labels_description_from_title, distances_description_from_title = p_description.knn_query(title_embedding_input, k=initial_k)

        # Borda Count cho tiêu đề từ tiêu đề
        for i in range(len(labels_title_from_title[0])):
            idx = labels_title_from_title[0][i]
            borda_scores[idx] += (0.7 * (initial_k - i))  # Trọng số cho tiêu đề

        # Borda Count cho mô tả từ tiêu đề
        for i in range(len(labels_description_from_title[0])):
            idx = labels_description_from_title[0][i]
            borda_scores[idx] += (0.4 * (initial_k - i))  # Trọng số cho mô tả

    # Nếu chỉ có tóm tắt
    elif abstract_input:
        abstract_embedding_input = model.encode([abstract_input])
        # Tìm labels và distances cho tóm tắt
        labels_title_from_abstract, distances_title_from_abstract = p_title.knn_query(abstract_embedding_input, k=initial_k)
        labels_description_from_abstract, distances_description_from_abstract = p_description.knn_query(abstract_embedding_input, k=initial_k)

        # Borda Count cho tiêu đề từ tóm tắt
        for i in range(len(labels_title_from_abstract[0])):
            idx = labels_title_from_abstract[0][i]
            borda_scores[idx] += (0.5 * (initial_k - i))  # Trọng số cho tiêu đề từ tóm tắt

        # Borda Count cho mô tả từ tóm tắt
        for i in range(len(labels_description_from_abstract[0])):
            idx = labels_description_from_abstract[0][i]
            borda_scores[idx] += (0.7 * (initial_k - i))  # Trọng số cho mô tả từ tóm tắt

    # Sắp xếp Borda Score
    sorted_borda = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = [data_reference.iloc[idx] for idx, _ in sorted_borda[:k]]

    # Lọc kết quả theo Rank và Field
    filtered_results = []
    for result in top_results:
        if (not ranks or result['Rank'] in ranks) and \
            (not selected_field or result['FoR1 Name'] == selected_field):
            result['Borda_Score'] = '{:.1f}'.format(borda_scores[result.name]) if result.name in borda_scores else '0.00'
            filtered_results.append(result)

    num_results = len(filtered_results)
    return render_template('results.html', results=filtered_results, num_results=num_results, title_input=title_input, abstract_input=abstract_input, ranks=ranks, selected_field=selected_field, borda_scores=borda_scores)

@app.route('/conference/<int:article_id>')
def conference_detail(article_id):
    # Lấy thông tin hội thảo từ cơ sở dữ liệu hoặc file dựa trên article_id
    conference = get_conference_by_id(article_id)  # Hàm giả định lấy dữ liệu

    if conference:
        return render_template('conference.html', conference=conference)
    else:
        # Nếu không tìm thấy hội thảo
        return render_template('404.html'), 404

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

if __name__ == '__main__':
    app.run(debug=True)
