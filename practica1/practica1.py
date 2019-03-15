import timeit
import argparse

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from relief import Relief
from local_search import LocalSearch


def pipeline(X, y, transformer, seed, make_trace):
    X = minmax_scale(X)
    accuracies = []
    times = []
    reductions = []
    traces = []
    kfold = KFold(5, shuffle=True, random_state=seed)
    for train_index, test_index in kfold.split(X):
        # Split training and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Beginning of timing zone
        t_start = timeit.default_timer()
        transformer.fit(X_train, y_train)
        t_stop = timeit.default_timer()
        # End of timing zone
        X_train = transformer.transform(X_train)
        X_test = transformer.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        # Gathering information
        acc = accuracy_score(knn.predict(X_test), y_test)
        accuracies.append(acc)
        times.append(t_stop - t_start)
        reductions.append(transformer.reduction)
        if make_trace:
            traces.append(transformer.trace)
    return np.array(accuracies), np.array(times), np.array(reductions), traces


def create_dataframe(accuracies, times, reductions):
    columns = ['Accuracy', 'Reduction', 'Aggregation', 'Time']
    index = ['Partition ' + str(i + 1) for i in range(5)]
    agreggations = (accuracies + reductions) / 2
    data = np.array([accuracies, reductions, agreggations, times]).T
    return pd.DataFrame(data=data, columns=columns, index=index)


def evaluate_algorithm(algorithm, X, Y, seed, make_trace=False):
    results = None
    if algorithm == 'relief':
        results = pipeline(X, Y, Relief(), seed, False)
    elif algorithm == 'local-search':
        results = pipeline(X, Y, LocalSearch(seed=seed), seed, make_trace)
    else:
        raise Exception('Please, use relief or local-search as parameters')
    return create_dataframe(*results[:-1]), results[-1]


def pretty_print(dataset, algorithm, seed, results):
    summary = results.describe().loc[['mean', 'std', '50%']]
    summary.index = ['Mean', 'Std.Dev', 'Median']
    output = """
=======================================================
    %s     |     %s      |  SEED = %d
=======================================================\n%s\n\n%s
    """ % (dataset.upper(), algorithm.upper(), seed, results.to_string(),
           summary)
    return output


def generate_graphics(filename, results, traces):
    _, axes = plt.subplots(2, len(results.columns) // 2, figsize=(10, 6))
    plt.suptitle('Algorithm Results', fontsize='x-large')
    for col, axis in zip(results.columns, axes.flatten()):
        results.boxplot(column=col, ax=axis)
    plt.savefig('%s_results.png' % filename)
    plt.clf()
    if traces:
        plt.title('Local Search fitness function trace')
        for i, t in enumerate(traces):
            plt.plot(t, label='Partition %d' % (i+1))
        plt.legend()
        plt.savefig('%s_trace.png' % filename)


def main(dataset, algorithm, seed, trace):
    df = pd.read_csv('./BIN/%s.csv' % dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    results, traces = evaluate_algorithm(algorithm, X, y, seed, trace)
    filename = 'output/%s_%s_%s' % (dataset, algorithm, seed)
    generate_graphics(filename, results, traces)
    output = pretty_print(dataset, algorithm, seed, results)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Values can be: ionosphere, colposcopy or texture')
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        help='Values can be: relief or local-search')
    parser.add_argument(
        '--trace',
        type=bool,
        required=False,
        default=False,
        help='Generate trace for local search? Values can be: True or False')
    parser.add_argument('--seed', type=int, required=True)
    args = vars(parser.parse_args())
    main(**args)
