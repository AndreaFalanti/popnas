from encoder import SearchSpace


def generate_legacy_dynamic_reindex_function(operators: 'list[str]', op_timers: 'dict[str, float]'):
    '''
    Closure for generating a function to easily apply dynamic reindex where necessary.
    LEGACY: POPNASv1 dynamic reindex, rewritten as a closure

    Args:
        operators: allowed operations
        op_timers: dict with op as key and time as value

    Returns:
        Callable[[str], float]: dynamic reindex function
    '''
    t_max = max(op_timers.values())

    def apply_dynamic_reindex(op_value: str):
        # in POPNASv1 dynamic reindex outputs in [0, len(operators)], so it is not normalized.
        return len(operators) * op_timers[op_value] / t_max

    return apply_dynamic_reindex


def build_legacy_feature_names(max_blocks: int):
    ''' LEGACY: POPNASv1 features, based mainly on cell specification and using categorical inputs. Has an extra "data_augmented" field. '''
    # create the complete headers row of the CSV files
    time_csv_headers = ['time', 'blocks']
    time_header_types, acc_header_types = ['Label', 'Num'], ['Label', 'Num']

    # create headers for csv files
    for b in range(1, max_blocks + 1):
        a = b * 2
        c = a - 1
        time_csv_headers.extend([f"input_{c}", f"operation_{c}", f"input_{a}", f"operation_{a}"])
        time_header_types.extend(['Categ', 'Num', 'Categ', 'Num'])
        acc_header_types.extend(['Categ', 'Categ', 'Categ', 'Categ'])

    # deep copy substituting first element (y column)
    # deep copy could be not necessary, but better than fighting with side-effect later on
    acc_csv_headers = ['acc'] + [header for header in time_csv_headers[1:]]

    # extra boolean field that simply state if the entry has been generated from data augmentation (False for original samples).
    # this field will be dropped during training, so it is not relevant for algorithms (catboost drops 'Auxiliary' header_type).
    time_csv_headers.append('data_augmented')
    time_header_types.append('Auxiliary')
    acc_csv_headers.append('data_augmented')
    acc_header_types.append('Auxiliary')

    return time_csv_headers, time_header_types, acc_csv_headers, acc_header_types


def generate_legacy_features(search_space: SearchSpace, current_blocks: int, time: float, accuracy: float, cell_spec: list):
    '''
    Builds all the allowed permutations of the blocks present in the cell, which are the equivalent encodings.
    Then, for each equivalent cell, produce the features set for both time and accuracy predictors.
    LEGACY: first tentative change to feature set, avoiding categorical inputs to allow the usage of more ML models and giving more info
     about NN structure. Use data augmentation (equivalent cell specifications) to make the model generalize on "positional features".

    Returns:
        (list): features to be used in predictors (ML techniques)
    '''

    # equivalent cells can be useful to train better the regressor
    eqv_cells, _ = search_space.generate_eqv_cells(cell_spec, size=search_space.B)

    # expand cell_spec for bool comparison of data_augmented field
    cell_spec = cell_spec + [(None, None, None, None)] * (search_space.B - current_blocks)

    return [[time, current_blocks] + search_space.encode_cell_spec(cell, op_enc_name='dynamic_reindex') + [cell != cell_spec]
            for cell in eqv_cells],\
           [[accuracy, current_blocks] + search_space.encode_cell_spec(cell) + [cell != cell_spec] for cell in eqv_cells]
