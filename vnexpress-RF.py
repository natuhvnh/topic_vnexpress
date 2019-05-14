import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

# load data
X = []
Y = []
label = ['cuoi', 'du-lich', 'gia-dinh', 'giai-tri', 'khoa-hoc', 'khoi-nghiep', 'kinh-doanh', 'oto-xe-may', 'phap-luat', 'so-hoa', 'suc-khoe',
        'tam-su', 'the-gioi', 'the-thao', 'thoi-su']
for i in range(0, len(label)):
    path = r'D:\NAL\Vnexpress\vnexpress\\' + label[i] + '\\' + label[i] + '.txt' # create path for each cate in for loop
    content = np.genfromtxt(path, delimiter = '\t', encoding= 'utf8', dtype= 'str') # read txt
    #content = np.reshape(content, (content.shape[0], 1)) # reshape in order to append 
    X.extend(content)
    Y.extend([label[i]] * len(content))

# convert content to tf-idf vector
vectorizer = TfidfVectorizer(ngram_range=(2,2))
X = vectorizer.fit_transform(X)
Label = LabelEncoder()
Y = np.asarray(Y)
Y = Label.fit_transform(Y.ravel())
# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 20)

# Random forest
model = RandomForestClassifier(n_estimators=5, random_state=0)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print(classification_report(Y_test,Y_predict))  
print(accuracy_score(Y_test, Y_predict))  

# accuracy = 65%
