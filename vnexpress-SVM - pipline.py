import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

# load data
X = []
Y = []
label = ['cuoi', 'du-lich', 'gia-dinh', 'giai-tri', 'khoa-hoc', 'khoi-nghiep', 'kinh-doanh', 'oto-xe-may', 'phap-luat', 'so-hoa', 'suc-khoe',
        'tam-su', 'the-gioi', 'the-thao', 'thoi-su']
for i in range(0, len(label)):
    path = r'/home/natu/natu/NAL/Vnexpress/vnexpress/' + label[i] + '/' + label[i] + '.txt' # create path for each cate in for loop
    content = np.genfromtxt(path, delimiter = '\t', encoding= 'utf8', dtype= 'str') # read txt
    #content = np.reshape(content, (content.shape[0], 1)) # reshape in order to append 
    X.extend(content)
    Y.extend([label[i]] * len(content))

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 20)

# USE (1) OR (2)
# (1): Dung gridsearch de test cac bo parameters => chon bo parameters co socre lon nhat. Thoi gian chay lau.
# Doc: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
"""
model = Pipeline([('vect', CountVectorizer(max_df= 0.5)),
                ('tfidf', TfidfTransformer(sublinear_tf = True)),
                ('clf', svm.LinearSVC(fit_intercept = True, multi_class='crammer_singer', C=1))])
parameters = {
            'vect__ngram_range': [(2, 2), (3, 3)],
            'tfidf__use_idf': (True, False),
            'clf__multi_class': ('crammer_singer', 'ovr')
            }
gs_model = GridSearchCV(model, parameters, iid=False, n_jobs=-1)
gs_model = gs_model.fit(X_train, Y_train)
print(gs_model.best_score_)
print(gs_model.cv_results_)
"""
# (2): Tune parameters thong thuong or dung parameters tu best score (1)
model = Pipeline([('vect', CountVectorizer(ngram_range= (2, 2), max_df= 0.5)),
                ('tfidf', TfidfTransformer(sublinear_tf = True)),
                ('clf', svm.LinearSVC(fit_intercept = True, multi_class='crammer_singer', C=1))])
model.fit(X_train, Y_train)
Y_predicted = model.predict(X_test)
report = metrics.classification_report(Y_test, Y_predicted)
accuracy = np.mean(Y_predicted == Y_test)
print(report)
print(accuracy)

# best score: 91.13%
