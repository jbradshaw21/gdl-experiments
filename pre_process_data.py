import numpy as np


def well_gen(letter_idx, num_idx):
    well_id = []
    for letter in letter_idx:
        for number in num_idx:
            well_str = letter + f"{number}"
            if len(well_str) == 2:
                well_str = well_str[0] + "0" + well_str[1]
            well_id.append(" " + well_str)

    return well_id


def normalise_features(data, features):

    new_data = {}
    for feature in features:

        feature_list = []
        max_element = max(data[feature])
        min_element = min([item for item in data[feature]])

        for data_idx, data_item in enumerate(data[feature], 0):
            data_item -= min_element
            data_item /= (max_element - min_element)

            # Fix Ch2 masks total + avg intensities to zero, if other Ch2 mask values are zero
            if feature in ['TotalIntenCh2', 'AvgIntenCh2']:
                if data["SpotCountCh2"][data_idx] == 0:
                    data_item = 0

            feature_list.append(data_item)

        new_data[feature] = feature_list

    return data


def pre_process_data(data, features, sample_type, model_used, data_set):

    if model_used == 'gnn' or model_used == 'attention_gnn' or model_used == 'cnn':
        data = normalise_features(data=data, features=features)

    # Find the cell count per well & insert into a dictionary
    well_list = []
    for idx, well in enumerate(data["WellId"], 0):
        if well not in well_list:
            well_list.append(well)

    # Generate well labels & insert into a dictionary
    well_4184 = well_gen(letter_idx=["C", "D", "E", "F"], num_idx=range(3, 15))
    well_4951 = well_gen(letter_idx=["G", "H", "I", "J"], num_idx=range(3, 15))
    well_1854 = well_gen(letter_idx=["K", "L", "M", "N"], num_idx=range(3, 15))
    well_OM_DMSO = well_gen(letter_idx=["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"], num_idx=[15])
    well_OM = well_gen(letter_idx=["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"], num_idx=[16])
    well_CM = well_gen(letter_idx=["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"], num_idx=[17])
    well_error = well_gen(letter_idx=["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"], num_idx=[18])

    starting_concentration = 10
    concentration_dict = {}
    for i in range(12):
        well_concentration = well_gen(letter_idx=["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"],
                                      num_idx=[i + 3])
        concentration = starting_concentration / (3 ** i)
        concentration_dict.update(dict.fromkeys(well_concentration, concentration))

    labels = [well_4184, well_4951, well_1854, well_OM_DMSO, well_OM, well_CM, well_error]
    well_id = [0, 1, 2, 3, 4, 5, "error"]

    label_dict = {}
    for idx, label in enumerate(labels, 0):
        label_dict.update(dict.fromkeys(label, well_id[idx]))

    # Remove error column
    for well in [key for key, value in label_dict.items() if value == "error"]:
        well_list.remove(well)

    # If ignore controls:
    for well in [key for key, value in label_dict.items() if value == 3 or value == 4 or value == 5]:
        well_list.remove(well)
        labels = labels[:3]

    label_dict = {}
    for idx, label in enumerate(labels, 0):
        label_dict.update(dict.fromkeys(label, well_id[idx]))

    cell_count_dict = {}
    well_list = [[well] * 9 for well in well_list]
    field_num = [[i + 1 for i in range(9)] for _ in range(len(well_list))]

    well_list = [well for wells in well_list for well in wells]
    field_num = [num for nums in field_num for num in nums]

    sample_list = list(zip(well_list, field_num))

    for sample_idx, sample in enumerate(sample_list, 0):
        cell_count = 0
        for well_idx, well in enumerate(data["WellId"], 0):
            if well == sample[0] and sample_type == 'wells':
                cell_count += 1
            elif well == sample[0] and data["FieldNumber"][well_idx] == sample[1]:
                cell_count += 1

        cell_count_dict[sample[0]] = cell_count

        if data_set == 'Field':
            if np.isnan(data['SpotCountCh2'][sample_idx]):
                sample_list.remove((data['WellId'][sample_idx], data['FieldNumber'][sample_idx]))
        elif data_set == 'Well':
            data["FieldNumber"] = data["PlateId/Barcode"]

    return data, sample_list, cell_count_dict, concentration_dict, label_dict
