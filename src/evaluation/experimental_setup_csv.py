import sys
sys.path.append('../../')

import os
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import numpy as np
import pandas as pd
import math

from argparse import ArgumentParser


from scipy.spatial.distance import cdist

from src.base.data_set import DataSet
from src.utils.data_functions import load_data
from src.utils.evaluation_functions import eval_perfs, misclf_rate
from src.utils.mathematical_functions import kernels
from src.classifier.parzen_window_classifier import PWC
from src.query_strategies.expected_probabilistic_active_learning import XPAL
from src.query_strategies.probabilistic_active_learning import PAL
from src.query_strategies.uncertainty_sampling import US
from src.query_strategies.expected_error_reduction import EER
from src.query_strategies.random_sampling import RS
from src.query_strategies.optimal_sampling import OS
from src.query_strategies.active_learnig_with_cost_embedding import ALCE
from src.query_strategies.query_by_committee import QBC

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from time import process_time

##
from copy import copy, deepcopy
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
##

def run(data_set, results_path, query_strategy, budget, test_ratio, kernel, bandwidth, seed, category_filter):
    """
    Run experiments to compare query selection strategies.
    Experimental results are stored in a .csv-file.

    Parameters
    ----------
    data_set: str
        Name of the data set.
    results_path: str
        Absolute path to directory for saving results.
    query_strategy: str
        Determines query strategy.
    budget: int
        Maximal number of labeled samples.
    test_ratio: float in (0, 1)
        Ratio of test samples.
    kernel: str
        Kernel used by Parzen window classifier.
    bandwidth: float | 'scott'
        Determines the bandwidth of the used kernel.
    seed: float
        Random seed.
    """
    # --------------------------------------------- LOAD DATA ----------------------------------------------------------
    ##
    if data_set == 'new_vect_additional':
        Xo, yo = load_data(data_set_name='new_vect')
        Xa, ya = load_data(data_set_name='new_vect_additional')
        X = np.vstack((Xo, Xa))
        y = np.hstack((yo, ya))
        print(X.shape, y.shape)
    else:
        try:
            X, y = load_data(data_set_name=data_set)
        except:
            print('Data set {} not found. Conducting alternative load'.format(data_set))
            # load x and y from a single csv
            text_data = pd.read_csv('../../data/{}.csv'.format(data_set))
            # text_data = text_data[text_data['x'].str.len() > 50] # apply filter to remove short texts
            X = text_data.iloc[:, :-1].values
            y = text_data.iloc[:, -1].values
            print(X.shape, y.shape)
    
    if category_filter != None:
        # combine x and y into a pandas df 
        category_reduction_df = pd.DataFrame(X)
        category_reduction_df['y'] = y
        
        # remove categories with less than category_filter samples
        category_reduction_df = category_reduction_df.groupby('y').filter(lambda x: len(x) > category_filter)
        
        # remove categories with more than category_filter samples
        # category_reduction_df = category_reduction_df.groupby('y').filter(lambda x: len(x) < category_filter)
        
        # remove the groceries category
        # category_reduction_df = category_reduction_df[category_reduction_df['y'] != 'Groceries']
        print(category_reduction_df['y'].value_counts())
        # split into x and y
        X = category_reduction_df.iloc[:, :-1].values
        y = category_reduction_df.iloc[:, -1].values

    is_categorical = data_set in ['monks', 'car', 'bankruptcy', 'tic', 'corral']
    is_text = data_set in ['reports-compendium', 'reports-mozilla', 'text_data_all', 'text_data_original', 'text_data_additional']
    if is_categorical:
        X = OneHotEncoder(sparse=False).fit_transform(X)
        print(X.shape[1])

    n_features = np.size(X, axis=1)
    classes, class_distribution = np.unique(y, return_counts=True)
    print('{}: {}, {}'.format(data_set, classes, class_distribution))
    n_classes = len(classes)

    # --------------------------------------------- STATISTICS ---------------------------------------------------------
    # define storage for performances
    perf_results = {}

    # define performance functions
    perf_funcs = {'error': misclf_rate}

    # ------------------------------------------- LOAD DATA ----------------------------------------------------
    print('seed: {}'.format(str(seed)))
    test_ratio = float(test_ratio) if test_ratio < 1.0 else int(test_ratio)

    if data_set == 'new_vect_additional':
        seed_copy = deepcopy(seed)
        Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, ya, test_size=test_ratio, random_state=seed)
        if test_ratio < 1:
            while not np.array_equal(np.unique(ya_train), np.unique(ya_test)):
                Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, ya, test_size=test_ratio, random_state=seed)
                seed += 1000
                print('new seed: {}'.format(seed))  
        
        seed = seed_copy    
        Xo_train, Xo_test, yo_train, yo_test = train_test_split(Xo, yo, test_size=test_ratio, random_state=seed)
        if test_ratio < 1:
            while not np.array_equal(np.unique(yo_train), np.unique(yo_test)):
                Xo_train, Xo_test, yo_train, yo_test = train_test_split(Xo, yo, test_size=test_ratio, random_state=seed)
                seed += 1000
                print('new seed: {}'.format(seed))      

        # combine/stack Xa and Xo and ya and yo 
        X_train = np.vstack((Xo_train, Xa_train))
        X_test = np.vstack((Xo_test, Xa_test))
        y_train = np.hstack((yo_train, ya_train))
        y_test = np.hstack((yo_test, ya_test))
    elif data_set in ['text_data_all', 'text_data_original', 'text_data_additional']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
        if test_ratio < 1:
            while not np.array_equal(np.unique(y_train), np.unique(y_test)):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
                seed += 1000
                print('new seed: {}'.format(seed))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
        if test_ratio < 1:
            while not np.array_equal(np.unique(y_train), np.unique(y_test)):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
                seed += 1000
                print('new seed: {}'.format(seed))

    budget_csv = budget
    if budget == 'all':
        budget = len(X_train)
    elif test_ratio < 1 and int(budget) > len(X) * (1 - test_ratio):
        budget = int(math.floor(len(X) * (1 - test_ratio)))
    else:
        budget = int(budget)

    print(budget)

    # --------------------------------------------- CSV NAMES ----------------------------------------------------------
    category_filter_name = 'cat_fltr_{}'.format(category_filter) if category_filter != None else ''
    csv_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(data_set, query_strategy, test_ratio, kernel, bandwidth, budget_csv, seed, category_filter_name)

    # ------------------------------------------ PREPROCESS DATA -------------------------------------------------------
    # standardize data
    if not is_categorical and not is_text:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # ------------------------------------------ VECORIZE HERE! --------------------------------------------------------
    if data_set in ['text_data_all', 'text_data_original', 'text_data_additional']:
        vect = TfidfVectorizer(analyzer="word", strip_accents="unicode", max_features=9000, stop_words="english")
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        X_train = X_train[0].to_numpy()
        X_test = X_test[0].to_numpy()
        X_train = vect.fit_transform(X_train)
        X_test = vect.transform(X_test)

        X_train = X_train.toarray()
        X_test = X_test.toarray()

        le = preprocessing.LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # set up data set
    data = DataSet(X_train)

    # compute bandwidth
    n_samples = len(X_train)
    if bandwidth == 'scott':
        bandwidth = np.power(n_samples, -1. / (n_features + 4))
    elif bandwidth == 'mean':
        if kernel == 'rbf':
            nominator = 2 * n_samples * n_features
            denominator = (n_samples - 1) * np.log((n_samples - 1) / ((np.sqrt(2) * 10 ** -6) ** 2))
        elif kernel == 'categorical':
            nominator = (np.sum(cdist(XA=X_train, XB=X_train, metric='hamming')))/(n_samples**2 - n_samples)
            denominator = np.log((n_samples - 1) / ((np.sqrt(2) * 10 ** -6) ** 2))
        else:
            nominator = 1
            denominator = 1
        bandwidth = np.sqrt(nominator / denominator)
    else:
        bandwidth = float(bandwidth)
    print('bandwidth: {}'.format(str(bandwidth)))

    # create classifier
    gamma = 0.5 * (bandwidth ** (-2))
    pwc = PWC(n_classes=n_classes, metric=kernel, gamma=gamma, random_state=seed)

    if 'pal' in query_strategy:
        params = query_strategy.split('-')
        alpha = float(params[1]) if len(params) == 2 else 1
        S = kernels(X_train, X_train, metric=kernel, gamma=gamma)
        if query_strategy.startswith('xpal'):
            selection_strategy = XPAL(data_set=data, n_classes=n_classes, S=S, alpha_c=alpha, alpha_x=alpha,
                                      random_state=seed)
        elif query_strategy.startswith('pal'):
            selection_strategy = PAL(data_set=data, S=S, n_classes=n_classes, alpha_c=alpha, m_max=2,
                                     random_state=seed)
    elif query_strategy in ['lc', 'sm', 'entropy']:
        pwc_cpy = deepcopy(pwc)
        selection_strategy = US(data_set=data, model=pwc_cpy, method=query_strategy, random_state=seed)
    elif query_strategy in ['log-loss', 'zero-one-loss']:
        S = kernels(X_train, X_train, metric=kernel, gamma=gamma)
        selection_strategy = EER(data_set=data, n_classes=n_classes, S=S, method=query_strategy, random_state=seed)
    elif query_strategy == 'random':
        selection_strategy = RS(data_set=data, random_state=seed)
    elif query_strategy == 'optimal':
        S = kernels(X_train, X_train, metric=kernel, gamma=gamma)
        selection_strategy = OS(data_set=data, S=S, y=y_train, random_state=seed)
    elif query_strategy == 'alce':
        C = 1 - np.eye(n_classes)
        selection_strategy = ALCE(data_set=data, base_regressor=KernelRidge(kernel='rbf', gamma=gamma), C=C,
                                  method=query_strategy, random_state=seed)
    elif query_strategy == 'qbc':
        pwc_cpy = deepcopy(pwc)
        selection_strategy = QBC(data_set=data, model=pwc_cpy, random_state=seed)
    else:
        raise ValueError(
            "'query_strategy' must be in [xpal, pal, lc, sm, entropy, log-loss, zero-one-loss, random, optimal, qbc, alce]")

    # ----------------------------------------- ACTIVE LEARNING CYCLE --------------------------------------------------
    labeled = []
    times = [0]
    choosen_samples = []
    choosen_samples_labels = []
    for b in range(budget):
        print("budget: {}".format(b))
        # evaluate performance
        if test_ratio != 1:
            eval_perfs(clf=pwc, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, perf_results=perf_results,
                       perf_funcs=perf_funcs)

        # select sample
        t = process_time()
        sample_id = selection_strategy.make_query()
        times.append(process_time() - t)
        print(sample_id)
        choosen_samples.append(sample_id[0])

        # update training data
        data.update_entries(sample_id, y_train[sample_id])
        choosen_samples_labels.append(y_train[sample_id][0])

        # update statistics
        labeled.append(sample_id[0])

        # retrain classifier
        pwc.fit(data.X_[labeled], data.y_[labeled])

    # export labeled data as csv
    # df = pd.DataFrame({'labeled': labeled})
    # relative_path = '{}/labeled_{}.csv'.format(results_path, csv_name)
    # path = os.path.join(abs_path, relative_path)
    # df.to_csv(path, index_label='index')

    # evaluate final performance
    if test_ratio != 1:
        eval_perfs(clf=pwc, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, perf_results=perf_results,
                   perf_funcs=perf_funcs)

    # get absolute path
    abs_path = os.path.abspath(os.path.dirname(__file__))

    # create csv file to save samples that were chosen
    if seed%100 == 7:
        df_samples = pd.DataFrame({'samples': choosen_samples, 'labels': choosen_samples_labels})
        relative_path = '{}/samples_{}.csv'.format(results_path, csv_name)
        path = os.path.join(abs_path, relative_path)
        df_samples.to_csv(path, index_label='index')

    # # plot confusion matrix
    # y_pred = pwc.predict(X_test)
    # y_labels = ['Atm', 'Beauty', 'Bills And Household', 'Car' ,'Children' ,'Consumer Goods',
    #             'Culture', 'Digital Services' ,'Drugstore', 'Electronics', 'Fashion',
    #             'Financial Services', 'Food And Drink' ,'Freetime', 'Groceries', 'Health',
    #             'House And Garden', 'Investments', 'Pets' ,'Professional Services',
    #             'Shopping Online', 'Sport', 'Travel']
    # # create csv with test and predicted labels
    # cm_data = pd.DataFrame({'test': y_test, 'pred': y_pred})
    # relative_path = '{}/test_pred_{}.csv'.format(results_path, csv_name)
    # path = os.path.join(abs_path, relative_path)
    # cm_data.to_csv(path, index_label='index')

    # cm = confusion_matrix(y_test, y_pred)
    # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # # save confusion matrix to as pdf
    # relative_path = '{}/cm_{}.pdf'.format(results_path, csv_name)
    # path = os.path.join(abs_path, relative_path)
    # category_bins = [i for i in range(len(y_labels))]
    # plt.xticks(category_bins, y_labels, rotation='vertical', fontsize=8)
    # plt.yticks(category_bins, y_labels, fontsize=8)
    # plt.savefig(path, bbox_inches='tight')
    # # plt.show()
    # plt.close()

    # store performance results
    print(f'mean time: {np.mean(times)}')
    perf_results['time'] = times
    df = pd.DataFrame(perf_results)
    relative_path = '{}/performances_{}.csv'.format(results_path, csv_name)
    path = os.path.join(abs_path, relative_path)
    df.to_csv(path, index_label='index')


def main():
    parser = ArgumentParser(description='Parameters of experimental setup')
    parser.add_argument('--data_set', type=str, default='text_data_all', help='name of data set, see data_set_ids.csv for more information, '
                                                                     'default=iris')
    parser.add_argument('--results_path', type=str, help='absolute path for saving results: default=../../results')
    parser.add_argument('--query_strategy', default='random', type=str, help='name of active learning strategy: '
                                                                             '[xpal-0.001, pal-1, lc, alce, '
                                                                             'zero-one-loss, qbc, optimal], default=xpal-0.001')
    parser.add_argument('--budget', default=100, help='number of active learning iterations: default=200')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='ratio of test samples: default=0.4')
    parser.add_argument('--kernel', type=str, default='cosine', help='kernel used by Parzen window classifier: '
                                                                  '[rbf, categorical, cosine], default=rbf')
    parser.add_argument('--bandwidth', default='mean', help='kernel bandwidth: default=mean')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducibility: default=0')
    parser.add_argument('--category_filter', type=int, default=None, help='min number of samples per category')
    args = parser.parse_args()
    run(data_set=args.data_set, results_path=args.results_path, query_strategy=args.query_strategy, budget=args.budget,
        test_ratio=args.test_ratio, kernel=args.kernel, bandwidth=args.bandwidth, seed=args.seed, category_filter=args.category_filter)


if __name__ == '__main__':
    main()

