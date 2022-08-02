import warnings
from collections import Counter
# Библиотеки для манипуляции с данными
import pandas as pd
import numpy as np
# Библиотеки для визуализации данных
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# Предобработка
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
# Для понижения размерности данных, с помощью выбора K наиболее влиятельных
#   признаков
from sklearn.feature_selection import SelectKBest
# Модели для обучения
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Настройка и оценка
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     RandomizedSearchCV, GridSearchCV,
                                     ShuffleSplit)
from sklearn.metrics import (recall_score, confusion_matrix, f1_score,
                             accuracy_score, precision_score, roc_curve,
                             roc_auc_score)
from sklearn.feature_selection import SelectKBest, chi2, f_classif
# Для балансировки классов
from imblearn.over_sampling import SMOTE
