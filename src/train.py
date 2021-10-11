import log_service
import plotter
from manager import NetworkManager
from controller import ControllerManager
from encoder import StateSpace

from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

import importlib.util
import csv
import statistics

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages


class Train:

    def __init__(self, blocks, children_max_size,
                 dataset, sets,
                 epochs, batch_size, learning_rate, filters, weight_reg,
                 cell_stacks, normal_cells_per_stack,
                 all_blocks_concat, pnas_mode,
                 checkpoint, restore):

        self._logger = log_service.get_logger(__name__)

        # search space parameters
        self.blocks = blocks
        self.children_max_size = children_max_size

        # dataset parameters
        self.dataset = dataset
        self.sets = sets

        # CNN models parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.filters = filters
        self.weight_reg = weight_reg
        self.concat_only_unused = not all_blocks_concat
        self.cell_stacks = cell_stacks
        self.normal_cells_per_stack = normal_cells_per_stack

        self.pnas_mode = pnas_mode

        # for restoring a run
        self.checkpoint = checkpoint
        self.restore = restore

        plotter.initialize_logger()

    def load_dataset(self):
        if self.dataset == "cifar10":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar10.load_data()
        elif self.dataset == "cifar100":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar100.load_data()
        # TODO: untested legacy code, not sure this is still working
        else:
            spec = importlib.util.spec_from_file_location("dataset", self.dataset)
            dataset = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset)
            (x_train_init, y_train_init), (x_test_init, y_test_init) = dataset.load_data()

        return (x_train_init, y_train_init), (x_test_init, y_test_init)

    def prepare_dataset(self, x_train_init, y_train_init):
        """Build a validation set from training set and do some preprocessing

        Args:
            x_train_init (ndarray): x training
            y_train_init (ndarray): y training

        Returns:
            list:
        """
        # normalize image RGB values into [0, 1] domain
        x_train_init = x_train_init.astype('float32') / 255.

        datasets = []
        # TODO: why using a dataset multiple times if sets > 1? Is this actually useful or it's possible to deprecate this feature?
        # TODO: splits for other datasets are actually not defined
        for i in range(0, self.sets):
            # TODO: take only 10000 images for fast training (one batch of cifar10), make it random in future?
            # limit = 10000
            # x_train_init = x_train_init[:limit]
            # y_train_init = y_train_init[:limit]

            # create a validation set for evaluation of the child models
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, test_size=0.1, random_state=0,
                                                                            stratify=y_train_init)

            if self.dataset == "cifar10":
                # cifar10
                y_train = to_categorical(y_train, 10)
                y_validation = to_categorical(y_validation, 10)

            elif self.dataset == "cifar100":
                # cifar100
                y_train = to_categorical(y_train, 100)
                y_validation = to_categorical(y_validation, 100)

            # TODO: logic is missing for custom dataset usage

            # pack the dataset for the NetworkManager
            datasets.append([x_train, y_train, x_validation, y_validation])

        return datasets

    def generate_and_train_model_from_spec(self, state_space: StateSpace, manager: NetworkManager, cell_spec: list):
        """
        Generate a model given the actions and train it to get reward and time

        Args:
            state_space (StateSpace): ...
            manager (NetworkManager): ...
            cell_spec (list): plain cell specification

        Returns:
            tuple: reward, timer, params, flops of trained CNN
        """
        # print the cell in a more comprehensive way
        state_space.print_cell_spec(cell_spec)

        # save model if it's the last training batch (full blocks)
        last_block_train = len(cell_spec) == self.blocks
        # build a model, train and get reward and accuracy from the network manager
        reward, timer, total_params, flops = manager.get_rewards(cell_spec, save_best_model=last_block_train)

        self._logger.info("Best accuracy reached: %0.6f", reward)
        self._logger.info("Training time: %0.6f", timer)
        # format is a workaround for thousands separator, since the python logger has no such feature 
        self._logger.info("Total parameters: %s", format(total_params, ','))
        self._logger.info("Total FLOPS: %s", format(flops, ','))

        return reward, timer, total_params, flops

    def generate_dynamic_reindex_function(self, operators, op_timers: 'dict[str, float]'):
        '''
        Closure for generating a function to easily apply dynamic reindex where necessary.

        Args:
            operators (list<str>): allowed operations
            op_timers (list<float>): timers for each block with same operations, in order

        Returns:
            Callable[[str], float]: dynamic reindex function
        '''
        t_max = max(op_timers.values())

        def apply_dynamic_reindex(op_value: str):
            # TODO: remove len(operators) to normalize in 0-1?
            return len(operators) * op_timers[op_value] / t_max

        return apply_dynamic_reindex

    def perform_initial_thrust(self, state_space: StateSpace, manager: NetworkManager):
        '''
        Build a starting point model with 0 blocks to evaluate the offset (initial thrust).

        Args:
            state_space (StateSpace): [description]
            manager (NetworkManager): [description]
        '''

        self._logger.info('Performing initial thrust with empty cell')
        _, timer, _, _ = self.generate_and_train_model_from_spec(state_space, manager, [])

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            data = [timer, 0]
            for _ in range(self.blocks):
                data.extend([0, 0, 0, 0])
            writer = csv.writer(f)
            writer.writerow(data)

    def write_overall_cnn_training_results(self, blocks, timers, rewards):
        with open(log_service.build_path('csv', 'training_overview.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)

            # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
            if f.tell() == 0:
                writer.writerow(['# blocks', 'avg training time(s)', 'max time', 'min time', 'avg val acc', 'max acc', 'min acc'])

            avg_time = statistics.mean(timers)
            max_time = max(timers)
            min_time = min(timers)

            avg_acc = statistics.mean(rewards)
            max_acc = max(rewards)
            min_acc = min(rewards)

            writer.writerow([blocks, avg_time, max_time, min_time, avg_acc, max_acc, min_acc])

    # TODO: not really necessary, even b=1 could use the more powerful generate_eqv_cells_encodings. Keep it or not?
    def generate_sliding_block_encodings(self, current_blocks: int, timer: float, cell_spec: list, state_space: StateSpace):
        '''
        Usable for cells with b = 1. Simply slide the block in different positions. Very fast since it doesn't need to
        build all possible permutations.

        Returns:
            (list): encoded cells with additional data, to write in csv (list of lists)
        '''

        encoded_cell = state_space.encode_cell_spec(cell_spec, op_enc_name='dynamic_reindex')
        csv_rows = []

        for i in range(current_blocks, self.blocks + 1):
            data = [timer, current_blocks]

            # slide block forward
            for _ in range(current_blocks, i):
                data.extend([0, 0, 0, 0])

            # add reindexed block encoding (encoded cell is actually a single block)
            data.extend(encoded_cell)

            # extend with empty blocks, if necessary
            for _ in range(i + 1, self.blocks + 1):
                data.extend([0, 0, 0, 0])

            csv_rows.append(data)

        return csv_rows

    def generate_eqv_cells_encodings(self, current_blocks: int, timer: float, cell_spec: list, state_space: StateSpace):
        '''
        Needed for cells with b > 1, compared to sliding blocks it also builds the allowed permutations of the blocks
        present in the cell.

        Returns:
            (list): encoded cells with additional data, to write in csv (list of lists)
        '''

        # equivalent cells can be useful to train better the regressor
        eqv_cells, _ = state_space.generate_eqv_cells(cell_spec, size=self.blocks)
        # encode cell spec, using dynamic reindex for operators, but keep it as a list of tuples
        return list(map(lambda cell: [timer, current_blocks] + state_space.encode_cell_spec(cell, op_enc_name='dynamic_reindex'), eqv_cells))

    def write_training_time(self, current_blocks: int, timer: float, cell_spec: list, state_space: StateSpace):
        '''
        Write on csv the training time, that will be used for regressor training.
        Use sliding blocks mechanism to multiple the entries.

        Args:
            current_blocks (int): [description]
            timer (float): [description]
            cell_spec (list): [description]
            state_space (StateSpace): [description]
        '''

        csv_rows = self.generate_sliding_block_encodings(current_blocks, timer, cell_spec, state_space) if current_blocks == 1 \
            else self.generate_eqv_cells_encodings(current_blocks, timer, cell_spec, state_space)

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

    def write_catboost_column_desc_file(self, header_types):
        with open(log_service.build_path('csv', 'column_desc.csv'), mode='w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(enumerate(header_types))

    def process(self):
        '''
        Main function, executed by run.py to start POPNAS algorithm.
        '''

        # create the complete headers row of the CSV files
        headers = ["time", "blocks"]
        header_types = ['Label', 'Num']
        # dictionary to store specular monoblock (-1 input) times for dynamic reindex
        op_timers = {}
        reindex_function = None

        # TODO: restore search space
        operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        # operators = ['identity', '3x3 dconv']

        if self.restore:
            starting_b = self.checkpoint  # change the starting point of B

            self._logger.info("Loading operator indexes!")
            with open(log_service.build_path('csv', 'reindex_op_times.csv')) as f:
                reader = csv.reader(f)
                for row in reader:
                    # row is [time, op]
                    op_timers[row[1]] = float(row[0])

            reindex_function = self.generate_dynamic_reindex_function(operators, op_timers)
        else:
            starting_b = 0

            # create headers for csv files
            for b in range(1, self.blocks + 1):
                a = b * 2
                c = a - 1
                headers.extend([f"input_{c}", f"operation_{c}", f"input_{a}", f"operation_{a}"])
                header_types.extend(['Categ', 'Num', 'Categ', 'Num'])

            # add headers
            with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

            self.write_catboost_column_desc_file(header_types)

        # construct a state space
        state_space = StateSpace(self.blocks, input_lookback_depth=-2, input_lookforward_depth=None, operators=operators)

        # print the state space being searched
        state_space.print_state_space()
        self._logger.info('Total cells stacked in each CNN: %d', (self.normal_cells_per_stack + 1) * self.cell_stacks - 1)
        self._logger.info('%s', '*' * 101)

        # load correct dataset (based on self.dataset), test data is not used actually
        (x_train_init, y_train_init), _ = self.load_dataset()

        dataset = self.prepare_dataset(x_train_init, y_train_init)

        # create the Network Manager
        manager = NetworkManager(dataset, data_num=self.sets, epochs=self.epochs, batchsize=self.batch_size,
                                 learning_rate=self.learning_rate, filters=self.filters, weight_reg=self.weight_reg,
                                 cell_stacks=self.cell_stacks, normal_cells_per_stack=self.normal_cells_per_stack,
                                 concat_only_unused=self.concat_only_unused)

        # create the ControllerManager and build the internal policy network
        controller = ControllerManager(state_space, self.checkpoint, B=self.blocks, K=self.children_max_size,
                                       train_iterations=15, reg_param=3e-5, lr1=0.002, controller_cells=60, embedding_dim=10,
                                       pnas_mode=self.pnas_mode, restore_controller=self.restore)

        # add dynamic reindex
        if self.restore:
            state_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)

        # if B = 0, perform initial thrust before starting actual training procedure
        if starting_b == 0:
            self.perform_initial_thrust(state_space, manager)
            starting_b = 1

        monoblock_times = []

        # train the child CNN networks for each number of blocks
        for current_blocks in range(starting_b, self.blocks + 1):
            rewards = []
            timers = []

            cell_specs = state_space.get_cells_to_train()

            for model_index, cell_spec in enumerate(cell_specs):
                self._logger.info("Model #%d / #%d", model_index + 1, len(cell_specs))
                self._logger.debug("\t%s", cell_spec)

                reward, timer, total_params, flops = self.generate_and_train_model_from_spec(state_space, manager, cell_spec)
                rewards.append(reward)
                timers.append(timer)

                if current_blocks == 1:
                    monoblock_times.append([timer, cell_spec])
                    # unpack the block (only tuple present in the list) into its components
                    in1, op1, in2, op2 = cell_spec[0]

                    # get required data for dynamic reindex
                    # op_timers will contain timers for blocks with both same operation and input -1, for each operation, in order
                    same_inputs = in1 == in2
                    same_op = op1 == op2
                    if same_inputs and same_op and in1 == -1:
                        with open(log_service.build_path('csv', 'reindex_op_times.csv'), mode='a+', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timer, op1])
                            op_timers[op1] = timer

                self._logger.info("Finished %d out of %d models!", (model_index + 1), len(cell_specs))

                # write the results of this trial into a file
                with open(log_service.build_path('csv', 'training_results.csv'), mode='a+', newline='') as f:
                    writer = csv.writer(f)

                    # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
                    if f.tell() == 0:
                        writer.writerow(['best val accuracy', 'training time(seconds)', 'total params', 'flops', '# blocks', 'cell structure'])

                    cell_structure = f"[{';'.join(map(lambda el: str(el), cell_spec))}]"
                    data = [reward, timer, total_params, flops, current_blocks, cell_structure]

                    writer.writerow(data)

                # in current_blocks = 1 case, we need all CNN to be able to dynamic reindex, so it is done outside the loop
                if current_blocks > 1:
                    self.write_training_time(current_blocks, timer, cell_spec, state_space)

            # current_blocks = 1 case, same mechanism but wait all CNN for applying dynamic reindex
            if current_blocks == 1:
                reindex_function = self.generate_dynamic_reindex_function(operators, op_timers)
                state_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)
                plotter.plot_dynamic_reindex_related_blocks_info()

                for timer, cell_spec in monoblock_times:
                    self.write_training_time(current_blocks, timer, cell_spec, state_space)

            self.write_overall_cnn_training_results(current_blocks, timers, rewards)

            # avoid controller training, pareto front estimation and plot at final step
            if current_blocks != self.blocks:
                loss = controller.train_step(rewards)
                self._logger.info("Trial %d: ControllerManager loss : %0.6f", current_blocks, loss)

                controller.update_step(headers)

                # remove invalid input values for current blocks
                inputs_to_prune_count = current_blocks + 1 - self.blocks
                valid_inputs = state_space.input_values if inputs_to_prune_count >= 0 else state_space.input_values[:inputs_to_prune_count]
                # PNAS mode doesn't build pareto front
                if not self.pnas_mode:
                    plotter.plot_pareto_inputs_and_operators_usage(current_blocks + 1, operators, valid_inputs)
                # state_space.children are updated in controller.update_step, CNN to train in next step
                plotter.plot_children_inputs_and_operators_usage(current_blocks + 1, operators, valid_inputs, state_space.children)

        plotter.plot_training_info_per_block()
        plotter.plot_cnn_train_boxplots_per_block(self.blocks)
        plotter.plot_predictions_error(self.blocks, self.pnas_mode)

        self._logger.info("Finished!")
