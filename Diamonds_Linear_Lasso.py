import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
from sklearn.preprocessing import StandardScaler#Data scaling
from sklearn.preprocessing import LabelEncoder#Label encoding
from sklearn import metrics
#Read the csv file.
diamonds = pd.read_csv('/Users/nikhil/Data/ML_examples/diamonds.csv')
variables = diamonds.columns
#Drop the unnamed columns
diamonds = diamonds.drop(['Unnamed: 0'], axis=1)
# Convert the categorical variables to numerical by label encoding. This can be done by writing a function that replaces the words in each categorical feature by a number. I use LabelEncoder to avoid extra lines of code.
categorical_features = ['cut', 'color', 'clarity']
le = LabelEncoder()
for i in range(3):
    new = le.fit_transform(diamonds[categorical_features[i]])
    diamonds[categorical_features[i]] = new
#Shorten the data set to improve execution time
diamonds=diamonds[0:5000]
#train-test split
Shuffle_diamonds = diamonds.sample(frac=1)
train_split = int(0.7*len(Shuffle_diamonds))
train_diamonds = Shuffle_diamonds[:train_split]
test_diamonds = Shuffle_diamonds[train_split:]
#Select X and y values
X_train = train_diamonds.drop('price',axis=1)
y_train = train_diamonds[['price']]
X_test = test_diamonds.drop('price',axis=1)
y_test = test_diamonds[['price']]
#Standardization. Change feature value to [(value-mean)/sigma]. This helps in increasing the accuracy of the model.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)
#Conver to arrays
y_train = np.array(y_train)[:,0]
y_test = np.array(y_test)[:,0]*1.0
#Add intercept term in X so that the intercept b is absorbed in coefficients: beta
X_train = np.c_[np.ones((np.shape(X_train)[0],1)),X_train]
X_test = np.c_[np.ones((np.shape(X_test)[0],1)),X_test]



#Our aim is now to minimize the cost function CF
#CF = sum(y-y_predicted)^2 + lambda(sum(beta))
#y_predicted = beta*x and lambda is the lagrange multiplier
#We will use Gradient Descent method, calculate the derivative of CF when beta is +ve/-ve and update weights
#Here I take the value of alpha to be 0.01. A more efficient way is to iterate over different values of alpha, calculate the r2 score and select the alpha for the best r2 score.
#Lasso regression using GD
epochs = 5000
alpha = 0.1
N = len(X_train)
lm = np.linspace(0.01,20,10)
beta_GD = np.zeros((np.shape(X_train)[1],len(lm)))
del_beta_GD = np.zeros((np.shape(X_train)[1],len(lm)))
CF= np.zeros((epochs,len(lm)))
r2_score = np.zeros(len(lm))
l=0
for l in range (0,len(lm)):
 lmda = lm[l]
 e=0
 for e in range (0,epochs):
        Y_pred = np.matmul(X_train,beta_GD[:,l])
        j=0
        for j in range(0,len(beta_GD[:,l])):
            if beta_GD[j,l]>0:
                del_beta_GD[j,l] = ((2/N)*(np.matmul(-X_train.T,(y_train-Y_pred)) + lmda))[j]
            else:
                del_beta_GD[j,l] = ((2/N)*(np.matmul(-X_train.T,(y_train-Y_pred)) - lmda))[j]
        CF[e,l] = (np.sum((y_train-Y_pred)**2)+lmda*np.sum(beta_GD[:,l]))/(2*len(y_train))
        beta_GD[:,l] = beta_GD[:,l] - alpha*del_beta_GD[:,l]
        Y_p = np.matmul(X_test,beta_GD[:,l])
        r2_score[l] = metrics.r2_score(y_test,Y_p)


