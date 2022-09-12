from sklearn.model_selection import train_test_split


def cross_validation(cv_folds, sample_list, label_dict, concentration_dict, cell_count_dict, sample_type):

    cv_samples = []
    for i in range(cv_folds - 1):

        if sample_type == 'wells':
            sample_list = [(sample[0], 0) for sample in sample_list]

        unique_sample_list = []
        for sample in sample_list:
            if sample not in unique_sample_list:
                unique_sample_list.append(sample)
        sample_list = unique_sample_list

        label_list = [label_dict[sample[0]] for sample in sample_list]
        concentration_list = [concentration_dict[sample[0]] for sample in sample_list]
        for sample_idx, sample in enumerate(sample_list, 0):
            if concentration_dict[sample[0]] > 10 / (3 ** 4):
                concentration_list[sample_idx] = 0
            elif concentration_dict[sample[0]] > 10 / (3 ** 8):
                concentration_list[sample_idx] = 10
            else:
                concentration_list[sample_idx] = 20

        cell_count_list = [cell_count_dict[sample[0]] for sample in sample_list]
        sorted_cell_count = list(set(cell_count_list))
        sorted_cell_count.sort()

        for sample_idx, sample in enumerate(sample_list, 0):
            if cell_count_dict[sample[0]] < sorted_cell_count[- int((2 / 3) * len(sorted_cell_count))]:
                cell_count_list[sample_idx] = 0
            elif cell_count_dict[sample[0]] < sorted_cell_count[- int((1 / 3) * len(sorted_cell_count))]:
                cell_count_list[sample_idx] = 100
            else:
                cell_count_list[sample_idx] = 200

        stratify_list = []
        for idx, _ in enumerate(sample_list, 0):
            # not enough samples to stratify by cell count, however since there are more fields this is okay
            if sample_type == 'wells':
                stratify_list.append(concentration_list[idx] + label_list[idx])
            else:
                stratify_list.append(concentration_list[idx] + cell_count_list[idx] + label_list[idx])

        sample_list, split_samples = train_test_split(sample_list, test_size=1 / (cv_folds - i),
                                                      stratify=stratify_list, random_state=42)

        cv_samples.append(split_samples)
    cv_samples.append(sample_list)

    return cv_samples
