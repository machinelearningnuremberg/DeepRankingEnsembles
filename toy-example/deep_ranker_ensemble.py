import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from task_sampler import get_batch, get_batch_val
from scipy.stats import norm

# Local repo imports parent folder path
import sys

# =======================================================================================
# Quick Configuration
non_transfer = False
transfer = True
# =======================================================================================

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

def average_ranks_deep_set(input, incumbent, rl_model, epsilon=10):

    input = (input[0], input[1], torch.cat((input[2], incumbent[None, :]), axis=0))

    score_list = []
    for sl in rl_model.forward_separate_deep_set(input):
        score_list += [sl.detach().numpy().flatten()]


    # Rank them and return the average rank.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(score_list, axis=-1)
    mean_rank = np.mean(ranks, axis=0)
    std_rank = np.sqrt(np.var(ranks, axis=0))

    best_y = mean_rank[-1]

    mean_rank = mean_rank[:-1]
    std_rank = std_rank[:-1]

    z = (mean_rank - best_y) / (std_rank + 1E-9)
    score = (mean_rank - best_y) * norm.cdf(z) + (std_rank + 1E-9) * norm.pdf(z)
    #score = mean_rank + 10*std_rank
    return score, mean_rank, std_rank, score_list

def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    device = y_pred.device
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    # Weighted ranking because it is more important to get the the first ranks right than the rest
    weight = np.log(np.arange(observation_loss.shape[-1]) + 2) # To prevent loss of log(1)
    weight = np.array(weight, dtype=np.float32)
    weight = torch.from_numpy(weight)[None, :].to(device)
    observation_loss = observation_loss / weight

    return torch.mean(torch.sum(observation_loss, dim=1))

def average_ranks(X_query, rl_model):
    # Calculating the average rank of all inputs.
    score_list = []
    for nn in rl_model.sc:
        score_list += [nn(X_query).detach().numpy().flatten()]

    # Rank them and return the average rank.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(score_list, axis=-1)
    mean_rank = np.mean(ranks, axis=0)

    return mean_rank




# Defining our ranking model as a DNN.
# Keeping the model simple for now.
class Scorer(nn.Module):
    # Output dimension by default is 1 as we need a real valued score.
    def __init__(self, input_dim=1, hidden_dim=16):
        super(Scorer, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def generate_loss(self, prediction, y_true):
        prediction, y_true = flatten_for_loss_list(prediction, y_true)
        # Viewing everything as a 2D tensor.
        y_true = y_true.view(-1, y_true.shape[-1])
        prediction = prediction.view(-1, prediction.shape[-1])
        loss = listMLE(prediction, y_true)
        return loss


def flatten_for_loss_list(pred, y):
    flatten_from_dim = len(pred.shape) - 2
    pred = torch.flatten(pred, start_dim=flatten_from_dim)
    y = torch.flatten(y, start_dim=flatten_from_dim)
    return pred, y


class DeepSet(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1, output_dim=1, hidden_dim=16):
        super(DeepSet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.rho = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),           
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Encoder: First get the latent embedding of the whole batch
        x = self.phi(x)

        # Pool operation: Aggregate all the outputs to a single output.
        #                 i.e across size of support set
        # Using mean as the validation error instead of sum
        # because the cardinality should be irrelevant
        x = torch.mean(x, dim=-2)

        # Decoder: Decode the latent output to result
        x = self.rho(x)

        return x


class RankingLossSurrogate(nn.Module):
    def __init__(self, input_dim, ssid, load=False, out_ds = 2, hidden_dim_scorer=16, hidden_dim_ds = 16, M = 10):
        super(RankingLossSurrogate, self).__init__()
        self.ssid = ssid
        self.M = M
        self.hidden_dim_scorer = hidden_dim_scorer
        self.hidden_dim_ds =  hidden_dim_ds
        self.incumbent = None

        self.out_ds = out_ds
        self.save_folder = "results/";
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        if load:
            self.load()
        else:
            self.input_dim = input_dim
            self.sc, self.ds_embedder = self.create_embedder_scorers_uncertainty(self.input_dim, self.hidden_dim_scorer)
        


    def create_embedder_scorers_uncertainty(self, in_dim, hidden_dim_scorer=16):
        ds_embedder = DeepSet(input_dim=in_dim + 1, latent_dim=8, output_dim=self.out_ds, hidden_dim=self.hidden_dim_ds)
        sc_list = []
        for i in range(self.M):
            sc_list += [Scorer(input_dim=self.out_ds + in_dim, hidden_dim = hidden_dim_scorer)]
        # For easing saving and loading from hard disk
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
        self.sc, self.ds_embedder = self.create_embedder_scorers_uncertainty(self.input_dim, self.hidden_dim_scorer)
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

    def generate_loss_DeepSet(self, prediction, y_true):
        prediction, y_true = flatten_for_loss_list(prediction, y_true)
        # Viewing everything as a 2D tensor.
        y_true = y_true.view(-1, y_true.shape[-1])
        prediction = prediction.view(-1, prediction.shape[-1])
        loss = listMLE(prediction, y_true)
        return loss

    def train_model_together(self, get_batch, get_batch_test, epochs, list_size, lr):
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])
        loss_list = []
        val_loss_list = []
        for _ in range(epochs):
            self.train()
            for __ in range(100):
                optimizer.zero_grad()

                s_ft_X, s_ft_y, q_ft_X, q_ft_y = get_batch( list_size)

                losses = []
                predictions = self.forward_separate_deep_set((s_ft_X, s_ft_y, q_ft_X))
                for p in predictions:
                    losses += [self.generate_loss_DeepSet(p, q_ft_y)]
                loss = torch.stack(losses).mean()

                loss.backward()
                optimizer.step()

            #val
            with torch.no_grad():
                s_ft_X, s_ft_y, q_ft_X, q_ft_y = get_batch_test( list_size)

                losses = []
                predictions = self.forward_separate_deep_set((s_ft_X, s_ft_y, q_ft_X))
                for p in predictions:
                    losses += [self.generate_loss_DeepSet(p, q_ft_y)]
                val_loss = torch.stack(losses).mean()
                

            print("Epoch[", _, "] ==> Loss =", loss.item(), "; Val_loss =", val_loss.item())
            loss_list += [loss.item()]
            val_loss_list += [val_loss.item()]

        return loss_list, val_loss_list




    def get_fine_tune_batch(self, X_obs, y_obs):

        # Taking 20% of the data as the support set.
        support_size = max(1,int(0.3 * X_obs.shape[0]))
        idx_support = np.random.choice(X_obs.shape[0], size=support_size, replace=False)
        idx_query = np.delete(np.arange(X_obs.shape[0]), idx_support)

        s_ft_X = X_obs[idx_support]
        s_ft_y = y_obs[idx_support]
        q_ft_X = X_obs[idx_query]
        q_ft_y = y_obs[idx_query]

        return s_ft_X, s_ft_y, q_ft_X, q_ft_y

    def fine_tune_single(self, nn, X_obs, y_obs, epochs, lr):
        epochs = epochs
        loss_list = []
        optimizer = torch.optim.Adam([{'params': nn.parameters(), 'lr': lr},])
        for i in range(epochs):
            nn.train()
            optimizer.zero_grad()

            prediction = nn.forward(X_obs)
            loss = nn.generate_loss(prediction, y_obs)
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

    def fine_tune(self, X_obs, y_obs, epochs, lr):
        for nn in self.sc:
            self.fine_tune_single(nn, X_obs, y_obs, epochs, lr)

    def forward(self, input):
        s_X, s_y, q_X = input

        # Creating an embedding of X:y for the support data using the embedder
        s_X = torch.cat((s_X, s_y), dim=-1)
        s_X = self.ds_embedder(s_X)

        # Creating an input for the scorer.
        s_X = s_X[..., None, :]
        repeat_tuple = (1,) * (len(s_X.shape)-2) + (q_X.shape[-2], 1)
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
        repeat_tuple = (1,) * (len(s_X.shape)-2) + (q_X.shape[-2], 1)
        s_X = s_X.repeat(repeat_tuple)
        q_X = torch.cat((s_X, q_X), dim=-1)

        predictions = []
        for s in self.sc:
            predictions += [s(q_X)]

        return predictions

    def fine_tune_together(self, X_obs, y_obs, epochs, lr):
        loss_list = []
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])

        if len(X_obs)>2:
            for i in range(epochs):
                self.train()
                optimizer.zero_grad()

                s_ft_X, s_ft_y, q_ft_X, q_ft_y = self.get_fine_tune_batch(X_obs, y_obs)

                losses = []
                predictions = self.forward_separate_deep_set((s_ft_X, s_ft_y, q_ft_X))
                for p in predictions:
                    losses += [self.generate_loss_DeepSet(p, q_ft_y)]
                loss = torch.stack(losses).mean()

                loss.backward()
                optimizer.step()

                loss_list += [loss.item()]
            print("Last loss:", loss)

if __name__ == "__main__":


    DRE= RankingLossSurrogate( 1, ssid="toy", out_ds=10, hidden_dim_scorer=10, hidden_dim_ds = 10)
    DRE.train_model_together(get_batch, get_batch_val, epochs=1000, list_size=5, lr=0.001)
    DRE.save()
