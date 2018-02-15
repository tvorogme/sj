import pickle
from collections import *
import collections
import numpy as np
import networkx as nx
from tqdm import tqdm
import json
from typing import List, Tuple, Sequence
import pickle
from tqdm import tqdm_notebook as tqdm
import pymorphy2
import nltk
from cytoolz import pipe
from collections import Counter
import re
import theano
from collections import defaultdict
import gensim
import plotly.plotly as py
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
api_key = 'Kyxdr2dZqCwvx2yfpTvd'
import plotly
plotly.tools.set_credentials_file(username='tvorogme', api_key=api_key)
k = pickle.load(open("p.pck", 'rb'))
vacancy_by_name = pickle.load(open("vacancy_by_name.pck", "rb"))
morph = pymorphy2.MorphAnalyzer()
stopwords = nltk.corpus.stopwords.words(
    'russian') + nltk.corpus.stopwords.words('english')
stopwords += [
    'отличный', 'метр', 'наш', 'клиент', 'банка', 'проект', 'литр',
    'желательный', 'др', 'самый', 'мочь', 'хороший', 'год', 'чел', 'обязательный', 'опыт', 'работа',
    'качества', 'работа', 'свои'
]
resume = pickle.load(open("resume_with_days.pkl", "rb"))

themes = defaultdict(lambda: [])
dirty_to_normal = {}

for d in vacancy_by_name:
    d = vacancy_by_name[d]
    themes[d['profession_tree_name']].append(d['dirty_skills'])
    
cache = {}
vacancy = pickle.load(open("vacancy_with_vecs.pck", "rb"))
themes_vectors = defaultdict(lambda: [])

from joblib import delayed, Parallel

def parallel(f, data):
    """Run parallel your func on all CPU"""
    
    return Parallel(n_jobs=-1, verbose=3, max_nbytes='1G')(delayed(f)(x) for x in data)

def get_jobs(resume_id):
    '''Отсортируем работы по времени. 1999, 2000, 2010'''
    ans = {}
    for job in resume[resume_id]['work_history']:
        ans[int(job['monthbeg'])/12 + int(job['yearbeg'])] = job
    return dict(collections.OrderedDict(sorted(ans.items())))

sim_to_work = pickle.load(open("sim_to_work.pck", "rb"))
for work in sim_to_work:
    if sim_to_work[work]:
        tmp = vacancy_by_name[sim_to_work[work][0][0]]

        if 'popularity' not in tmp:
            tmp['popularity'] = 0

        tmp['popularity'] += 1
similar_to_theme = {}

for theme in tqdm(themes):
    cool = []
    for name in vacancy_by_name:
        tmp = vacancy_by_name[name]
        if 'popularity' in tmp and tmp['profession_tree_name'] == theme:
            cool.append([tmp['popularity'], tmp['id']])
    cool.sort(reverse=True)
    similar_to_theme[theme] = cool[:10]

def career(jobs):
    '''Текущая карьера по работе'''
    jobs = get_jobs(jobs)
    ans = list()
    for key in jobs:
        _id = jobs[key]['id']
        if _id in sim_to_work:
            ans.append(vacancy_by_name[sim_to_work[_id][0][0]]['profession_tree_name'])
            
    return ans


data = parallel(career, range(len(resume)))
data.sort(key=len)

normal_data = []

for i in data[::-1]:
    normal_data.append(i)

# matrix[i][j] - кол-во человек, которое из должности i перешли в должности j
matrix_back = defaultdict(lambda: Counter())

for line in normal_data:
    if len(line) > 1:
        first = line[0::2]
        second = line[1::2]
        
        for i in range(len(second)):
            if k[first[i]] < k[second[i]]:
                matrix_back[first[i]][second[i]] += 1
                
                
for vac in vacancy:
    if not np.isnan(vac['vec']).all():
        themes_vectors[vac['profession_tree_name']].append(vac['vec'])
       
for theme in themes_vectors:
    themes_vectors[theme] = np.sum(themes_vectors[theme], axis=0)

def compile_cos_sim_theano():
    v1 = theano.tensor.vector(dtype='float32')
    v2 = theano.tensor.vector(dtype='float32')
    
    numerator = theano.tensor.sum(v1*v2)
    denominator = theano.tensor.sqrt(theano.tensor.sum(v1**2)*theano.tensor.sum(v2**2))
   
    return theano.function([v1, v2], numerator/denominator)

cos_sim_theano_fn = compile_cos_sim_theano()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def _get_POS(word):
    if word not in cache:
        cache[word] = morph.parse(word)[0]
    return cache[word].tag.POS


def normal_form(word):
    if word not in cache:
        cache[word] = morph.parse(word)[0]
    return cache[word].normal_form


def is_word_pos_in(word: str, pos: List[str] = None) -> bool:
    if not pos:
        pos = ['NOUN', "ADJF", 'INFN', 'VERB', 'ADJS']

    return _get_POS(word) in pos


def get_words(text):
    return re.findall(r'\w+', text)


def nonempty(x):
    if isinstance(x, Sequence):
        return filter(lambda x: len(x) > 0 and x != ' ', x)
    return x


helper = {}


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def normalize_skill(skill: str):
    parsed = tuple(
        pipe(
            skill,
            lambda x: x.lower(),
            remove_numbers,
            get_words,
        ))

    clear_skill = []
    dirty_skill = []

    # Последнее стоп слово для dirty_skill
    last_stopword = None

    # Для каждого слова в скилле
    for i in parsed:
        # Нормализуем слово
        word = normal_form(i)

        # Если стоп слово - запомним его
        if word in nltk.corpus.stopwords.words('russian'):
            last_stopword = word

            if word == "без":
                clear_skill.append(word)

        # Проверим на часть речи, длинну и стоплова
        elif is_word_pos_in(word) and len(word) > 3 and word not in stopwords:

            # Если до этого было стоп слово, добавим его в dirty
            if last_stopword and len(dirty_skill) > 0:
                dirty_skill.append(last_stopword)
                last_stopword = None

            # Добавим в чистый скилл слово
            clear_skill.append(word)

            if is_word_pos_in(word, ['NOUN', 'ADJF']):
                dirty_skill.append(i)

    if len(dirty_skill) > 2 and len(dirty_skill) < 8:
        return clear_skill, dirty_skill

    else:
        return []

model = gensim.models.KeyedVectors.load_word2vec_format("ruwikiruscorpora_upos_cbow_300_20_2017.bin.gz", binary=True)
tmp_vector = np.array([0] * 300, dtype=np.float32)

def _word2vec(word):
    for i in ["_NOUN", "_ADJ", "_VERB"]:
        tmp = "{}{}".format(word, i)
        
        if tmp in model:
            return model[tmp]
        else:
            return tmp_vector

def to_vec(tupl):
    skills_vecs = []
    for skill in tupl:
        
        skill_vec = np.asarray(list(map(lambda x: _word2vec(morph.parse(x)[0].normal_form), skill)))
        if not np.array_equal(tmp_vector, skill_vec):
            skills_vecs.append(np.mean(skill_vec, axis=0))
    return np.mean(skills_vecs, axis=0)

def get_theme(tupl):
    vec = to_vec(tupl)
    a = Counter()
    
    for theme in themes_vectors:
        a[theme] = cos_sim_theano_fn(themes_vectors[theme], vec)
    
    for u in a.most_common(10):
        if k[u[0]] > 2: 
            return u[0]

def recommend_vac(_next, cur_skills):
    '''Рекоммендуем вакансию по скиллам'''
    current_vec = to_vec(cur_skills)
    vecs = [to_vec(vacancy_by_name[i[1]]['clear_skills']) for i in similar_to_theme[_next]]

    vecs = list(filter(lambda x: not np.isnan(x).all(), vecs))

    sim = list(map(lambda x: float(cos_sim_theano_fn(x, current_vec)), vecs))
    vacs = [vacancy_by_name[i[1]]['profession'] for i in similar_to_theme[_next]]

    return Counter({x:y for x,y in zip(vacs, sim)}).most_common(3)


def draw(theme, cur_skills= None):
    a = nx.DiGraph()

    def get_all_ages(prof, used=None):
        global k
        if not used:
            used = [prof]
        else:
            used.append(prof)

        now = []

        # {'Генеральный директор': 19}
        for i in matrix_back[prof].most_common():
            if i[0] not in used:  # False
                rating_next = k[i[0]]  # 10
                rating_now = k[prof]  # 9

                if rating_next > rating_now and rating_now + 2 >= rating_next:
                    now.append(i[0])

                    if len(now) > 2:
                        break
        for j in now:
            if prof and prof not in a:
                a.add_node(prof)

            if j and j not in a:
                a.add_node(j)

            if j:
                a.add_edge(prof, j)

            get_all_ages(j, used)

    get_all_ages(theme)
    p = nx.random_layout(a)

    new_p = {}
    prev_level_count = defaultdict(lambda: 0)

    for node in a.nodes():
        # (Название, x, y)

        i = [0, 0]
        i[0] = k[node] + 0.03 * prev_level_count[k[node]]
        i[1] = prev_level_count[k[node]]
        prev_level_count[k[node]] += 1
        new_p[node] = i

    p = new_p
    
    def _arrow(x, y, x2, y2):
        return dict(
            ax=x2,
            ay=y2,
            axref='x',
            ayref='y',
            x=x,
            y=y,
            xref='x',
            yref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=1,
            arrowcolor='#636363'
        )
    arrows = []
    for edge in a.edges():
        x0, y0 = p[edge[0]]
        x1, y1 = p[edge[1]]

        arrows.append(_arrow(x1, y1, x0, y0))

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        showlegend = True,
        marker=Marker(
            showscale=True,
            reversescale=True,
            colorscale="YIGnBu",
            color=[],
            size=20))
    x_text, y_text, labels = [], [], []
    
    for node in a.nodes():
        x, y = p[node]
        x_text.append(x)
        y_text.append(y)
        
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        
        if len(node.split()) > 1:
            a = node.split()
            a = a[:len(a)//2] + ["<br>"] + a[len(a)//2:]
            labels.append(" ".join(a))
        else:
            labels.append(node)
            
        if cur_skills:
            if node == theme:
                node_trace['text'].append(None)
            else:
                t = []
                for i in recommend_vac(node, cur_skills):
                    
                    if len(i[0].split()) > 1:
                        a = i[0].split()
                        a = a[:len(a)//2] + ["<br>"] + a[len(a)//2:]
                        t.append(" ".join(a))
                    else:
                        t.append(i[0])

                    
                
                node_trace['text'].append("<br><br>".join(t))
        
        if node == theme:
            node_trace['marker']['color'].append("red")
        else:
            node_trace['marker']['color'].append(k[node])

    trace2 = Scatter(
        x=x_text,
        y=y_text,
        mode='markers+text',
        name='Markers and Text',
        text=labels,
        textposition='top',
        hoverinfo='none',
        showlegend = False,
        marker=Marker(color=[],size=0)
    )

    data = [node_trace, trace2]
    layout = Layout(showlegend=False, annotations=arrows)
    fig = Figure(data=data, layout=layout)
    return fig