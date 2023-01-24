import os
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import argparse

# Local functionality imports
from ranking_losses import get_ranking_loss
from acquisitions import get_acuisition_func
from HPO_B.hpob_handler import HPOBHandler
from utility import store_object, get_input_dim, convert_meta_data_to_np_dictionary
from utility import get_all_combinations, evaluate_combinations

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
result_folder = None

def flatten_for_loss(pred, y):
    flatten_from_dim = len(pred.shape) - 2
    pred = torch.flatten(pred, start_dim=flatten_from_dim)
    y = torch.flatten(y, start_dim=flatten_from_dim)
    return pred, y


def generate_loss(prediction, y_true):
    prediction, y_true = flatten_for_loss(prediction, y_true)
    # Viewing everything as a 2D tensor.
    y_true = y_true.view(-1, y_true.shape[-1])
    prediction = prediction.view(-1, prediction.shape[-1])

    f = get_ranking_loss(parser.parse_args().loss_func)
    loss = f(prediction, y_true)

    return loss


def get_fine_tune_batch_DeepSet(X_obs, y_obs):
    # Taking 20% of the data as the support set.
    support_size = int(0.2 * X_obs.shape[0])
    idx_support = np.random.choice(X_obs.shape[0], size=support_size, replace=False)
    idx_query = np.delete(np.arange(X_obs.shape[0]), idx_support)

    s_ft_X = X_obs[idx_support]
    s_ft_y = y_obs[idx_support]
    q_ft_X = X_obs[idx_query]
    q_ft_y = y_obs[idx_query]

    return s_ft_X, s_ft_y, q_ft_X, q_ft_y


def get_batch_HPBO_DeepSet(meta_data, batch_size, list_size, random_state=None):
    support_X = []
    support_y = []
    query_X = []
    query_y = []

    rand_num_gen = np.random.RandomState(seed=random_state)

    # Sample all tasks and form a high dimensional tensor of size
    #   (tasks, batch_size, list_size, input_dim)
    for data_task_id in meta_data.keys():
        data = meta_data[data_task_id]
        X = data["X"]
        y = data["y"]
        idx_support = rand_num_gen.choice(X.shape[0], size=(batch_size, 20), replace=True)
        support_X += [torch.from_numpy(X[idx_support])]
        support_y += [torch.from_numpy(y[idx_support])]
        idx_query = rand_num_gen.choice(X.shape[0], size=(batch_size, list_size), replace=True)
        query_X += [torch.from_numpy(X[idx_query])]
        query_y += [torch.from_numpy(y[idx_query])]

    return torch.stack(support_X), torch.stack(support_y), torch.stack(query_X), torch.stack(query_y)


def get_batch_HPBO_single_DeepSet(meta_train_data, list_size):
    support_size = int(0.2 * list_size)
    data = meta_train_data[np.random.choice(list(meta_train_data.keys()))]
    support_X = []
    support_y = []
    query_X = []
    query_y = []
    X = data["X"]
    y = data["y"]
    if support_size > X.shape[0] // 2:
        support_size = X.shape[0] // 2
    idx_support = np.random.choice(X.shape[0], size=support_size, replace=False)
    support_X += [torch.from_numpy(X[idx_support])]
    support_y += [torch.from_numpy(y[idx_support])]

    query_choice = np.setdiff1d(np.arange(X.shape[0]), idx_support, assume_unique=False)
    if list_size > X.shape[0] - support_size:
        list_size = X.shape[0] - support_size
    if list_size > query_choice.shape[0]:
        list_size = query_choice.shape[0]

    idx_query = np.random.choice(query_choice, size=list_size, replace=False)

    query_X += [torch.from_numpy(X[idx_query])]
    query_y += [torch.from_numpy(y[idx_query])]
    return torch.stack(support_X), torch.stack(support_y), torch.stack(query_X), torch.stack(query_y)


def get_batch_HPBO(meta_data, batch_size, list_size, random_state=None):
    query_X = []
    query_y = []

    rand_num_gen = np.random.RandomState(seed=random_state)  # As of now unused

    # Sample all tasks and form a high dimensional tensor of size
    #   (tasks, batch_size, list_size, input_dim)
    for data_task_id in meta_data.keys():
        data = meta_data[data_task_id]
        X = data["X"]
        y = data["y"]
        idx_query = rand_num_gen.choice(X.shape[0], size=(batch_size, list_size), replace=True)
        query_X += [torch.from_numpy(X[idx_query])]
        query_y += [torch.from_numpy(y[idx_query][..., 0])]

    return torch.stack(query_X), torch.stack(query_y)


def get_batch_HPBO_single(meta_train_data, batch_size, slate_length):
    query_X = []
    query_y = []
    for i in range(batch_size):
        data = meta_train_data[np.random.choice(list(meta_train_data.keys()))]
        X = data["X"]
        y = data["y"]
        idx = np.random.choice(X.shape[0], size=slate_length, replace=True)
        query_X += [torch.from_numpy(X[idx])]
        query_y += [torch.from_numpy(y[idx].flatten())]
    return torch.stack(query_X), torch.stack(query_y)


# Defining our scoring model as a DNN.
class Scorer(nn.Module):
    # Output dimension by default is 1 as we need a real valued score.
    def __init__(self, input_dim=1):
        super(Scorer, self).__init__()
        self.input_dim = input_dim

        # Creating the required neural networks with RELU activation function.
        n_h_layers = parser.parse_args().layers - 1

        p = (nn.Linear(input_dim, 32), nn.ReLU(),)
        for _ in range(n_h_layers):
            p = p + (nn.Linear(32, 32), nn.ReLU(),)
        p = p + (nn.Linear(32, 1),)

        self.model = nn.Sequential(*p)

    def forward(self, x):
        x = self.model(x)
        return x

    def meta_train(self, meta_train_data, meta_val_data, epochs, batch_size, list_size, lr):
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])  # 0.0001 giving good results
        loss_list = []
        val_loss_list = []
        for _ in range(epochs):
            self.train()
            for __ in range(100):
                optimizer.zero_grad()

                train_X, train_y = get_batch_HPBO_single(meta_train_data, 1, list_size)
                prediction = self.forward(train_X)
                loss = generate_loss(prediction, train_y)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.eval()

                # Calculating full training loss
                train_X, train_y = get_batch_HPBO(meta_train_data, batch_size, list_size)
                prediction = self.forward(train_X)
                loss = generate_loss(prediction, train_y)

                # Calculating validation loss
                val_X, val_y = get_batch_HPBO(meta_val_data, batch_size, list_size)
                pred_val = self.forward(val_X)
                val_loss = generate_loss(pred_val, val_y)

            print("Epoch[", _, "] ==> Loss =", loss.item(), "; Val_loss =", val_loss.item())
            loss_list += [loss.item()]
            val_loss_list += [val_loss.item()]

        return loss_list, val_loss_list


class DeepSet(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1, output_dim=1):
        super(DeepSet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.phi = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.rho = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # Encoder: First get the latent embedding of the whole batch
        x = self.phi(x)

        # Pool operation: Aggregate all the outputs to a single output.
        #                 i.e across size of support set
        # Using mean as the validation error instead of sum
        # because the model should be agnostic to any given support set cardinality
        x = torch.mean(x, dim=-2)

        # Decoder: Decode the latent output to result
        x = self.rho(x)

        return x


class DeepRankingEnsemble(nn.Module):
    def __init__(self, input_dim, ssid, M, loading=False):
        super(DeepRankingEnsemble, self).__init__()
        self.ssid = ssid
        self.M = M
        self.loading = loading
        self.incumbent = None

        self.save_folder = result_folder
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        if self.loading:
            self.load()
        else:
            self.input_dim = input_dim
            self.sc, self.ds_embedder = self.create_embedder_scorers_uncertainty(self.input_dim, self.M)

    def create_embedder_scorers_uncertainty(self, in_dim, M):
        ds_embedder = nn.Identity()
        sc_list = []
        for i in range(M):
            sc_list += [Scorer(input_dim=in_dim)]
        # Re-structure our module if deep set is enabled.
        if parser.parse_args().meta_features:
            ds_embedder = DeepSet(input_dim=in_dim + 1, latent_dim=32, output_dim=16)
            sc_list = []
            for i in range(M):
                sc_list += [Scorer(input_dim=16 + in_dim)]
        # Using Module List for easing saving and loading from hard disk
        return nn.ModuleList(sc_list), ds_embedder

    def save(self):
        file_name = self.save_folder + self.ssid
        state_dict = self.sc.state_dict()
        ds_embedder_state_dict = self.ds_embedder.state_dict()
        torch.save({"input_dim": self.input_dim,
                    "ssid": self.ssid,
                    "M": self.M,
                    "scorer": state_dict,
                    "ds_embedder": ds_embedder_state_dict,
                    "save_folder": self.save_folder},
                   file_name)

    def load(self):
        file_name = self.save_folder + self.ssid
        state_dict = torch.load(file_name)
        dict = torch.load(file_name)
        self.input_dim = dict["input_dim"]
        self.ssid = dict["ssid"]
        self.M = dict["M"]
        self.save_folder = dict["save_folder"]

        # Creating and initializing the scorer and embedder
        self.sc, self.ds_embedder = self.create_embedder_scorers_uncertainty(self.input_dim, self.M)
        self.sc.load_state_dict(state_dict["scorer"])
        self.ds_embedder.load_state_dict(state_dict["ds_embedder"])

    def train_model_separate(self, meta_train_data, meta_val_data, epochs, batch_size, list_size, lr):
        loss_list = []
        val_loss_list = []
        for nn in self.sc:
            l, vl = nn.meta_train(meta_train_data, meta_val_data, epochs, batch_size, list_size, lr)
            loss_list += [l]
            val_loss_list += [vl]

        loss_list = np.array(loss_list, dtype=np.float32)
        val_loss_list = np.array(val_loss_list, dtype=np.float32)
        loss_list = np.mean(loss_list, axis=0).tolist()
        val_loss_list = np.mean(val_loss_list, axis=0).tolist()

        return loss_list, val_loss_list


    def train_model_together(self, meta_train_data, meta_val_data, epochs, batch_size, list_size, lr):
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])
        loss_list = []
        val_loss_list = []
        for _ in range(epochs):
            self.train()
            for __ in range(100):
                optimizer.zero_grad()

                s_ft_X, s_ft_y, q_ft_X, q_ft_y = get_batch_HPBO_single_DeepSet(meta_train_data, list_size)

                losses = []
                predictions = self.forward_separate_deep_set((s_ft_X, s_ft_y, q_ft_X))
                for p in predictions:
                    losses += [generate_loss(p, q_ft_y)]
                loss = torch.stack(losses).mean()

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.eval()

                # Calculating full training loss
                s_ft_X, s_ft_y, q_ft_X, q_ft_y = get_batch_HPBO_DeepSet(meta_train_data, batch_size, list_size)
                losses = []
                predictions = self.forward_separate_deep_set((s_ft_X, s_ft_y, q_ft_X))
                for p in predictions:
                    losses += [generate_loss(p, q_ft_y)]
                loss = torch.stack(losses).mean()

                # Calculating validation loss
                s_ft_X, s_ft_y, q_ft_X, q_ft_y = get_batch_HPBO_DeepSet(meta_val_data, batch_size, list_size)
                losses = []
                predictions = self.forward_separate_deep_set((s_ft_X, s_ft_y, q_ft_X))
                for p in predictions:
                    losses += [generate_loss(p, q_ft_y)]
                val_loss = torch.stack(losses).mean()

            print("Epoch[", _, "] ==> Loss =", loss.item(), "; Val_loss =", val_loss.item())
            loss_list += [loss.item()]
            val_loss_list += [val_loss.item()]

        return loss_list, val_loss_list

    def train_model_separate(self, meta_train_data, meta_val_data, epochs, batch_size, list_size, lr):
        loss_list = []
        val_loss_list = []
        for nn in self.sc:
            l, vl = nn.meta_train(meta_train_data, meta_val_data, epochs, batch_size, list_size, lr)
            loss_list += [l]
            val_loss_list += [vl]

        loss_list = np.array(loss_list, dtype=np.float32)
        val_loss_list = np.array(val_loss_list, dtype=np.float32)
        loss_list = np.mean(loss_list, axis=0).tolist()
        val_loss_list = np.mean(val_loss_list, axis=0).tolist()

        return loss_list, val_loss_list

    def fine_tune_single(self, nn, X_obs, y_obs, epochs, lr):
        epochs = epochs
        loss_list = []
        optimizer = torch.optim.Adam([{'params': nn.parameters(), 'lr': lr}, ])
        for i in range(epochs):
            nn.train()
            optimizer.zero_grad()

            prediction = nn.forward(X_obs)
            loss = generate_loss(prediction, y_obs)
            loss.backward()

            optimizer.step()
            loss_list += [loss.item()]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss for listwise Ranking loss"]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.ssid + "_fine_tune_loss.png")
        plt.close()

    def fine_tune_separate(self, X_obs, y_obs, epochs, lr):
        for nn in self.sc:
            self.fine_tune_single(nn, X_obs, y_obs, epochs, lr)

    # The difference between forward and forward_separate_deep_set is in the
    # returned output.
    #     forward - Returns mean of the predicted scores.
    #     forward_separate_deep_set - Returns the list of scores predicted by
    #                                 the neural networks in the ensemble.
    def forward(self, input):
        s_X, s_y, q_X = input

        # Creating an embedding of X:y for the support data using the embedder
        s_X = torch.cat((s_X, s_y), dim=-1)
        s_X = self.ds_embedder(s_X)

        # Creating an input for the scorer.
        s_X = s_X[..., None, :]
        repeat_tuple = (1,) * (len(s_X.shape) - 2) + (q_X.shape[-2], 1)
        s_X = s_X.repeat(repeat_tuple)
        q_X = torch.cat((s_X, q_X), dim=-1)

        predictions = []
        for s in self.sc:
            predictions += [s(q_X)]

        predictions = torch.stack(predictions)
        return torch.mean(predictions, dim=0)

    def forward_separate_deep_set(self, input):
        s_X, s_y, q_X = input

        # Creating an embedding of X:y for the support data using the embedder
        s_X = torch.cat((s_X, s_y), dim=-1)
        s_X = self.ds_embedder(s_X)

        # Creating an input for the scorer.
        s_X = s_X[..., None, :]
        repeat_tuple = (1,) * (len(s_X.shape) - 2) + (q_X.shape[-2], 1)
        s_X = s_X.repeat(repeat_tuple)
        q_X = torch.cat((s_X, q_X), dim=-1)

        predictions = []
        for s in self.sc:
            predictions += [s(q_X)]

        return predictions

    def fine_tune_together(self, X_obs, y_obs, epochs, lr):
        epochs = epochs
        loss_list = []
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])
        for i in range(epochs):
            self.train()
            optimizer.zero_grad()

            s_ft_X, s_ft_y, q_ft_X, q_ft_y = get_fine_tune_batch_DeepSet(X_obs, y_obs)

            losses = []
            predictions = self.forward_separate_deep_set((s_ft_X, s_ft_y, q_ft_X))
            for p in predictions:
                losses += [generate_loss(p, q_ft_y)]
            loss = torch.stack(losses).mean()

            loss.backward()
            optimizer.step()

            loss_list += [loss.item()]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss for listwise Ranking loss"]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.ssid + "_" +
                    str(parser.parse_args().eval_index) + "_fine_tune_loss.png")
        plt.close()

    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        X_obs = torch.from_numpy(X_obs)
        y_obs = torch.from_numpy(y_obs)
        X_pen = torch.from_numpy(X_pen)

        if self.incumbent is None:
            inc_idx = np.argmax(y_obs)
            self.incumbent = X_obs[inc_idx]

        cli_args = parser.parse_args()

        learning_rate = 0.001
        if not self.loading:
            # A slightly higher learning rate for non transfer case.
            learning_rate = 0.02

        # Doing reloads from the saved model for every fine tuning.
        # For non transfer case loading = false ==> DRE randomly initialized.
        restarted_model = DeepRankingEnsemble(input_dim=self.input_dim,
                                              ssid=self.ssid,
                                              M=self.M,
                                              loading=self.loading)
        if cli_args.meta_features:
            restarted_model.fine_tune_together(X_obs, y_obs, epochs=1000, lr=learning_rate)
        else:
            restarted_model.fine_tune_separate(X_obs, y_obs, epochs=1000, lr=learning_rate)

        f = get_acuisition_func(cli_args.acq_func, cli_args.meta_features)
        scores = f((X_obs, y_obs, X_pen), self.incumbent, restarted_model)
        idx = np.argmax(scores)
        self.incumbent = X_pen[idx]

        return idx


def evaluate_keys(hpob_hdlr, keys_to_evaluate):
    performance = []

    cli_args = parser.parse_args()

    loading = not cli_args.non_transfer

    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = get_input_dim(hpob_hdlr.meta_test_data[search_space])
        method = DeepRankingEnsemble(input_dim=input_dim,
                                     ssid=search_space,
                                     M=cli_args.M,
                                     loading=loading)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res

    return performance


def evaluate_search_space_id(i):
    hpob_hdlr = HPOBHandler(root_dir="./HPO_B/hpob-data/", mode="v3-test")
    keys = get_all_combinations(hpob_hdlr, 100)
    print("Evaluating", i, "of ", len(keys))
    keys = keys[i:i + 1]  # Only executing the required key.
    performance = evaluate_keys(hpob_hdlr, keys_to_evaluate=keys)
    store_object(performance, result_folder + "/EVAL_KEY_" + str(i))


def meta_train_on_HPOB(i):
    hpob_hdlr = HPOBHandler(root_dir="./HPO_B/hpob-data/", mode="v3")

    # Pretrain Ranking loss surrogate with a single search space
    for search_space_id in sorted(hpob_hdlr.get_search_spaces())[i:i + 1]:
        t_start = time.time()

        meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
        meta_val_data = hpob_hdlr.meta_validation_data[search_space_id]

        input_dim = get_input_dim(meta_train_data)
        print("Input dim of", search_space_id, "=", input_dim)

        meta_train_data = convert_meta_data_to_np_dictionary(meta_train_data)
        meta_val_data = convert_meta_data_to_np_dictionary(meta_val_data)

        cli_args = parser.parse_args()

        epochs = 5000
        batch_size = 100
        list_size = 100
        lr = cli_args.lr_training

        rl_surrogate = DeepRankingEnsemble(input_dim=input_dim,
                                           ssid=search_space_id,
                                           M=cli_args.M,
                                           loading=False)
        if cli_args.meta_features:
            loss_list, val_loss_list = \
                rl_surrogate.train_model_together(meta_train_data, meta_val_data, epochs, batch_size, list_size, lr)
        else:
            loss_list, val_loss_list = \
                rl_surrogate.train_model_separate(meta_train_data, meta_val_data, epochs, batch_size, list_size, lr)

        rl_surrogate.save()
        rl_surrogate.load()

        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        plt.plot(np.array(val_loss_list, dtype=np.float32))
        legend = ["Loss",
                  "Validation Loss"
                  ]
        plt.legend(legend)
        plt.title("SSID: " + search_space_id + "; Input dim: " + str(input_dim))
        plt.savefig(rl_surrogate.save_folder + "loss_" + search_space_id + ".png")

        t_end = time.time()
        print("SSID:", search_space_id, "Completed in", t_end - t_start, "s")


if __name__ == '__main__':
    # Setting the command line options first
    parser.add_argument("--train", action="store_true",
                        help="Specify this to train the DRE.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Specify this to evaluate the DRE.")
    parser.add_argument("--train_index", type=int, default=0,
                        help="Index of the search space to train [0-15]."
                             " Only for transfer mode.")
    parser.add_argument("--eval_index", type=int, default=0,
                        help="Index of key to evaluate [0-429].")
    parser.add_argument("--non_transfer", action="store_true",
                        help="Specify this to run a non-transfer version of DRE.")
    parser.add_argument("--acq_func", type=str, default="ei",
                        help="Acquisition function to use during BO iteration ['avg', 'ucb', 'ei'].")
    parser.add_argument("--loss_func", type=str, default="listwise-weighted",
                        help="Ranking loss to use ['listwise-weighted', "
                             "'listwise', 'pairwise', 'pointwise'].")
    parser.add_argument("--lr_training", type=float, default=0.001,
                        help="The learning rate for the meta-training.")
    parser.add_argument("--meta_features", action="store_true", default=False,
                        help="Switch to enable the use of meta-features which is obtained by using deep set in the model.")
    parser.add_argument("--layers", type=int, default=4,
                        help="The number of layers in the neural network.")
    parser.add_argument("--M", type=int, default=10,
                        help="The number of neural networks in the ensemble.")
    parser.add_argument("--result_folder", type=str, default="./results/",
                        help="Folder where all result files are stored.")
    args = parser.parse_args()

    result_folder = args.result_folder

    if args.non_transfer:
        prefix = "DRE Non Transfer:"
    else:
        prefix = "DRE Transfer:"

    if args.train and not args.non_transfer:
        print(prefix, "Meta-training", args.train_index)
        meta_train_on_HPOB(args.train_index)

    if args.evaluate:
        print(prefix, "Evaluating", args.eval_index)
        evaluate_search_space_id(args.eval_index)
