### Introduction
The goal of this notebook is to provide some technical details (key steps) for our submitted manuscript: *Haolin Wang, et al. "Integrating Co-clustering and Interpretable Machine Learning for the Prediction of Intravenous Immunoglobulin Resistance in Kawasaki Disease", 2020.* 
To improve the performance of clinical prediction models addressing the incompleteness of EHRs data, the proposed method performed co-clustering to characterize the availability of clinical data, group Lasso for group-based feature selection, and Explainable Boosting Machine for group-specific prediction in a sequential manner.



### Tools and Implementation

#### Important packages

- Popular Python machine learning libraries such as scikit-learn, pandas and numpy

- imbalanced-learn: https://github.com/scikit-learn-contrib/imbalanced-learn

- InterpretML: https://github.com/interpretml/interpret

- Coclust: https://pypi.org/project/coclust/

- Group Lasso: https://github.com/AnchorBlues/GroupLasso/blob/master/grouplasso/model.py

#### Preprocessing

```
import pandas as pd

dat = pd.read_csv('dataset.csv')
df = pd.get_dummies(dat, columns=['categories'])
```

#### Over-sampling for imbalanced dataset

```
from imblearn.over_sampling import SMOTE

balanced_feature_set, balanced_label = SMOTE(sampling_strategy=<>, random_state=0).fit_resample(feature, label)
```

#### Missing data imputation to address data missing at random

```
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=5, random_state=0, tol=0.005)
feature_set = imp.fit_transform(feature_set)
```

#### Dataset splitting for cross-validation

```
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)
```

#### Baseline methods

```
#Lasso
from sklearn.linear_model import Lasso
#Logistic
from sklearn.linear_model import LogisticRegression
#Ridge
from sklearn.linear_model import Ridge
#KNN
from sklearn import neighbors
model = neighbors.KNeighborsClassifier()
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
#MLP
from sklearn.neural_network import MLPClassifier
#Random forest
from sklearn.ensemble import RandomForestClassifier
#lightGBM
import lightgbm 
#XGBoost
import xgboost as xgb
#EBM
from interpret.glassbox import ExplainableBoostingClassifier


```

#### Tuning the hyper-parameters

```
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

params = {"alpha": numpy.logspace(-3, 1, 5)}
model_cv = GridSearchCV(Lasso(), params, cv=5)
model = model_cv.fit(train_set, train_label)
print("tuned hpyerparameters:", model.best_params_)
prob = model_cv.predict(test_set)
```



### Co-clustering and Classification

#### Indicator matrix for incomplete dataset

```
# generate indicator matrix using the original dataset without missing data imputation
# 
[rows, cols] = train_set_original.shape
train_ind = numpy.zeros((rows, cols))
for r in range(rows):
	for c in range(cols):
		if not numpy.isnan(train_set_original[r, c]):
			train_ind[r, c] = 1
train_rows_num = rows

...

# merge training set and test set splitted for cross-validation
full_ind = numpy.row_stack((train_ind, test_ind))
```

#### Co-clustering of the indicator matrix 

```
from coclust.coclustering import CoclustInfo

for row_cluster in range(2, 10):
	for column_cluster in range(2,30):
		...
        model = CoclustInfo(n_row_clusters=row_cluster,n_col_clusters=column_cluster, random_state=42)
        model.fit(full_ind)
        row_labels = model.row_labels_
        print(row_labels)
        col_labels = model.column_labels_
        print(col_labels)
        ...
```

#### Re-organize samples for each row clusters to train multiple classifiers

```
for cluster_index in range(0, row_cluster):
	co_train_set = []
	co_train_label = []
	for k in range(0, train_rows_num):
		if row_labels[k] == cluster_index:
			co_train_label.append(train_label[k])
			co_train_set.append(train_set[k,:])
	co_train_set = numpy.array(co_train_set)
	co_train_label = numpy.array(co_train_label)
	print(co_train_set.shape)
	print(co_train_label.shape)
```

#### Feature selection with group feature structure

Group Lasso derives feature coefficients from certain groups to be small or exact zero. 

```
# train group lasso
from grouplasso import GroupLassoClassifier

model = GroupLassoClassifier(group_ids=numpy.array(col_labels),alpha=alpha_test, eta=0.001, tol=0.001, max_iter=3000, random_state=42, verbose_interval=300)
model.fit(co_train_set, co_train_label)
print(model.coef_)

# feature selection
selected_cols = []
for k in range(0, len(col_labels)):
	if abs(model.coef_[k]) > 0:
		selected_cols.append(k)

# train group-specific prediction model
decision_model = ExplainableBoostingClassifier().fit(co_train_set[:, selected_cols], co_train_label)
predict_test += list(decision_model.predict_proba(co_test_set[:, selected_cols])[:,1])
predict_test_label += list(co_test_label)
```

