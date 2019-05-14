import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras.regularizers import l2

# load data
label = ['cuoi', 'du-lich', 'gia-dinh', 'giai-tri', 'khoa-hoc', 'khoi-nghiep', 'kinh-doanh', 'oto-xe-may', 'phap-luat', 'so-hoa', 'suc-khoe',
        'tam-su', 'the-gioi', 'the-thao', 'thoi-su']
data = np.empty([0, 2])
for i in range(0, len(label)):
    path = r'/home/natu/natu/NAL/Vnexpress/vnexpress/' + label[i] + '/' + label[i] + '.txt' # create path for each cate in for loop
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
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X[:, 0])
X = X.toarray() # convert each row to tf-idf vector. Shape: 7285, 21443

# one hot vector label
encoder = LabelBinarizer()
Y = encoder.fit_transform(Y) # shape: 7275, 15

# split data
n = X.shape[0]
split = 0.8
X_train = X[0:round(n*split), :] # shape: 5092, 21443
X_test = X[round(n*split) : X.shape[0], :] # shape: 2183, 21443
Y_train = Y[0:round(n*split), :] # shape: 5092, 15
Y_test = Y[round(n*split) : Y.shape[0], :] # shape: 2183, 15
print(Y_train.shape)

# softmax regression
model = Sequential() 
model.add(Dense(c, activation='softmax', activity_regularizer= l2(0.001)))
optimizer = optimizers.Adam(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=20, batch_size=32, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)
print(score)





