import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

# 1 Load data set
data_set_cancer_initial = pd.read_csv('./kag_risk_factors_cervical_cancer.csv')

# 2 Replace the ? values for NaN
data_set_cancer_initial = data_set_cancer_initial.replace('?', np.NaN)

# We use a loop for trying different combinations that may better fit our model.
best_candidate = None
aquracy = 0
# 3
for ns in [('norm', preprocessing.MinMaxScaler()), ('srd', preprocessing.StandardScaler())]:
    # 4
    for classif in [('clf_3', GaussianNB()), ('clf_1', KNeighborsClassifier()), ('clf_2', DecisionTreeClassifier()), ('clf_4', SVC())]:

        # 5 Generate the pipe
        pipe_lr = Pipeline(steps=[ns, classif])

        # 6 We split the data and the label corresponding to the entire dataset
        training_data = data_set_cancer_initial.loc[:, data_set_cancer_initial.columns != 'Dx:Cancer']
        label_training = data_set_cancer_initial.loc[:, data_set_cancer_initial.columns == 'Dx:Cancer']

        # 7 Depending on the model to apply, we should select a range of the fitting hyper parameters.
        # KNN
        if classif[0] == 'clf_1':
            param_grid = {'clf_1__n_neighbors': list(range(1, 30)), 'clf_1__p': [2, 3, 4, 5]}

        # Decission Tree
        elif classif[0] == 'clf_2':
            param_grid = {'clf_2__criterion': ['gini', 'entropy'], 'clf_2__splitter': ['best', 'random']}

        # Gaussian Tree
        elif classif[0] == 'clf_3':
            param_grid = {}

        # SVM
        else:
            param_grid = {'clf_4__C': np.arange(1, 3), 'clf_4__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'clf_4__degree':
                np.arange(2, 8)}

        # 8
        sI = SimpleImputer(missing_values=np.nan, strategy='mean')
        sample = sI.fit(training_data)

        training_data = sample.transform(training_data)

        # 9 We construct the Grid search with corresponding configuration.
        time1 = time.time()
        grid_search = GridSearchCV(pipe_lr, param_grid, cv=5)
        # We fit our model in order to search the better configuration.
        grid_search.fit(training_data, label_training.values.ravel())

        time2 = time.time()

        # 10 Print the results
        print(grid_search.best_estimator_.steps, grid_search.best_score_)
        print("Training time for the model is {0}s".format(time2 - time1))

        if grid_search.best_score_ > aquracy:
            aquracy = grid_search.best_score_
            best_candidate = grid_search.best_estimator_.steps

# 11

pipe_new = Pipeline(steps=best_candidate)
print("The best configuration found is {0}".format(pipe_new))
CV = 10

# 13
results = cross_val_score(pipe_new, training_data, label_training.values.ravel(), cv=CV)
print('The average of the accuracy is: {0}'.format(np.mean(results)))

# 14
print("Visualize outliers")
sns.boxplot(data=pd.DataFrame(training_data))
plt.show()

# 15
# The next step is balancing the data controlled
# Now we take for instance the portion whose accuracy is better.
best_port = np.max(results)

ind = np.argmax(best_port)
ind += 1

# Now we calculate the size of the testing portion
tra_port = np.round(len(training_data) / CV)
end_pont = int(tra_port * ind)
beg_pont = int(end_pont - tra_port)

# With the best configuration and higher testing portion accuracy we have got
# We divide between the training and testing.
testing_fields = training_data[beg_pont:end_pont]
testing_label = label_training[beg_pont:end_pont]

training_field = np.concatenate((training_data[:beg_pont], training_data[end_pont+1:]), axis=0)
training_label = np.concatenate((label_training[:beg_pont], label_training[end_pont+1:]), axis=0)
# 16
# Tomek links
tl = TomekLinks()
X1, y1 = tl.fit_resample(training_field, training_label)

# Smote technique
sm = SMOTE(random_state=0)
X2, y2 = sm.fit_sample(training_field, training_label)

# Now we check which resampling model is better
pipe_new.fit(X1, y1)
predictedResults = pipe_new.predict(testing_fields)
a1 = accuracy_score(predictedResults, testing_label)
print("Accuracy Tomek: {0}".format(a1))

cf_mat = confusion_matrix(y_true=testing_label, y_pred=predictedResults)
print(cf_mat)

pipe_new.fit(X2, y2)
predictedResults = pipe_new.predict(testing_fields)
a2 = accuracy_score(predictedResults, testing_label)
print("Accuracy SMOTE: {0}".format(a2))

cf_mat = confusion_matrix(y_true=testing_label, y_pred=predictedResults)
print(cf_mat)

# 17
if a1 < best_port and a2 < best_port:
    print("It's better not to resample the training samples")

    best_training_fields = training_field
    best_label = training_label

else:
    if a1 < a2:
        print("Sampling with SMOTE is better")
        best_training_fields = X2
        best_label = y2
    else:
        print("Sampling with Tomek Links is better")
        best_training_fields = X1
        best_label = y1

# Now we have resampled or not, depending on the accuracy got.
# The last step is feature selection
# 18
selector = SelectPercentile(f_classif, percentile=25)
selector.fit(training_data, label_training)

# We get the importance of each feature for the model chosen.
print("Select percentil")
for n, s in zip(range(len(training_data[0])), selector.scores_):
    print("Score : ", s, " for feature ", n)

