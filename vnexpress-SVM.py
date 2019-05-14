import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split

# load data
label = ['cuoi', 'du-lich', 'gia-dinh', 'giai-tri', 'khoa-hoc', 'khoi-nghiep', 'kinh-doanh', 'oto-xe-may', 'phap-luat', 'so-hoa', 'suc-khoe',
        'tam-su', 'the-gioi', 'the-thao', 'thoi-su']
data = np.empty([0, 2])
for i in range(0, len(label)):
    path = r'D:\NAL\Vnexpress\vnexpress\\' + label[i] + '\\' + label[i] + '.txt' # create path for each cate in for loop
    content = np.genfromtxt(path, delimiter = '\t', encoding= 'utf8', dtype= 'str') # read txt
    content = np.reshape(content, (content.shape[0], 1)) # reshape in order to append 
    cate = np.chararray((content.shape[0], 1), itemsize= 20, unicode= True) # create array with string character
    cate[:] = label[i] # assign each row with label (cuoi, du lich ...)
    content = np.append(content, cate, 1)
    data = np.append(data, content, axis = 0)

# shuffle data
np.random.seed(0) #fix result of np.random
np.random.shuffle(data)
X = data[:, [0]] # shape: 7275, 1
Y = data[:, [1]] # shape: 7275, 1
d = X.shape[1] # number of features
c = len(np.unique(Y)) # number of classes
del data, content


# convert content to tf-idf vector
vectorizer = TfidfVectorizer(ngram_range=(2,2))
X = vectorizer.fit_transform(X[:, 0])
Label = LabelEncoder()
Y = Label.fit_transform(Y.ravel())

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 5)
"""
n = X.shape[0]
split = 0.8
X_train = X[0:round(n*split), :] # shape: 5092, 21443
X_test = X[round(n*split) : X.shape[0], :] # shape: 2183, 21443
Y_train = Y[0:round(n*split)] # shape: 5092,
Y_test = Y[round(n*split) : Y.shape[0]] # shape: 2183,
"""

# SVM

# # SVM with linear kernel
# C = 1 # svm regularization parameters
# SVM = svm.SVC(kernel= 'linear', C = C).fit(X_train, Y_train)
# accuracy = SVM.score(X_test, Y_test)
# print(accuracy)

# Linear SVM
C = 1 # svm regularization parameters
lin_svm = svm.LinearSVC(C= C).fit(X_train, Y_train)
accuracy = lin_svm.score(X_test, Y_test)
print(accuracy)

# # SVM with RBF (Radial basis function) kernel
# C = 1 # svm regularization parameters
# rbf_svm = svm.SVC(C= C, kernel= 'rbf', gamma= 0.06).fit(X_train, Y_train)
# accuracy = rbf_svm.score(X_test, Y_test)
# print(accuracy)

# SVM with polynomial RBF
# C = 1 # svm regularization parameters
# poly_svm = svm.SVC(C= C, kernel= 'poly', degree= 2).fit(X_train, Y_train)
# accuracy = poly_svm.score(X_test, Y_test)
# print(accuracy)


# SVM with linear kernel: Extremely slow
# Linear SVM: 91.13%
# SVM with RBF (Radial basis function) kernel: Extremely slow
# SVM with polynomial RBF: Extremely slow
