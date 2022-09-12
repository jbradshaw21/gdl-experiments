from matplotlib import pyplot as plt
import pandas as pd
from os.path import exists


def box_plot(cv_acc, experiment):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.boxplot(cv_acc, showmeans=True, meanprops=dict(marker='x', markeredgecolor='b'))
    plt.xticks(range(1, len(experiment) + 1), experiment)
    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Boxplot')
    plt.show()


for model_type in ['GNN', 'Log Reg', 'SVM', 'RF']:

    for experiment_type in ['Feature Selection', 'Hyper-parameter Tuning (ball query)', 'Hyper-parameter Tuning (knn)',
                            'Batch Size']:

        read_files = True
        experiment_no, read_fails = [0 for _ in range(2)]
        cv_acc_list, experiment_list = [[] for _ in range(2)]
        while read_files:
            experiment_no += 1
            path_to_file = f'C:/Users/jbrad/OneDrive/Documents/Thesis/Experiment Results/{model_type}/' \
                           f'{experiment_type}/Experiment {experiment_no}/training_out.csv'
            if exists(path_to_file):
                training_out = pd.read_csv(path_to_file)

                prev_fold = 1
                cv_accuracy = []
                for training_idx, training_fold in enumerate(training_out['fold'], 0):
                    if training_fold != prev_fold:
                        cv_accuracy.append(training_out['test_acc'][training_idx - 1])
                        prev_fold = training_fold
                cv_accuracy.append(training_out['test_acc'][len(training_out) - 1])

                cv_acc_list.append(cv_accuracy)
                experiment_list.append(experiment_no)
            else:
                read_fails += 1
                if read_fails > 10:
                    read_files = False
        if len(cv_acc_list) > 0:
            print(model_type, experiment_type)
            box_plot(cv_acc=cv_acc_list, experiment=experiment_list)
