import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
# Local imports

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

def acquisition_EI(X, y, X_query, rl_model):
    # https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
    # Find the best value of the objective function so far according to data.
    # Is this according to the gaussian fit or according to the actual values.???
    # For now using the best according to the actual values.
    best_y = np.max(y.detach().cpu().numpy())
    # Calculate the predicted mean & variance values of all the required samples
    mean, variance = rl_model.forward_mean_var((X, y, X_query))
    mean = mean.detach().cpu().numpy()
    std_dev = torch.sqrt(variance).detach().cpu().numpy()
    z = (mean - best_y) / (std_dev + 1E-9)
    return (mean - best_y) * norm.cdf(z) + (std_dev + 1E-9) * norm.pdf(z)

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
    weight = torch.from_numpy(weight)[None, :]
    observation_loss = observation_loss / weight

    return torch.mean(torch.sum(observation_loss, dim=1))

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

        # Pool: Agregate all the outputs to a single output. (i.e accross size of support set)
        # x = torch.sum(x, dim=-2)
        x = torch.mean(x, dim=-2)
        # Using mean as the validation error is jumping too much
        # Also because the cardinality should be irrelevant

        # Decoder: Decode the latent output to result
        x = self.rho(x)

        return x

# Defining our ranking model as a DNN.
# Keeping the model simple for now.
class Scorer(nn.Module):
    # Output dimension by default is 1 as we need a real valued score.
    def __init__(self, input_dim=1):
        super(Scorer, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.model(x)
        # The last layer must not be passed through relu

        # however we pass it through torch.tanh to keep the output in a resonable range
        # I had answered regarding this on stackoverflow
        # https://ai.stackexchange.com/questions/31595/are-the-q-values-of-dqn-bounded-at-a-single-timestep/31648#31648
        # The range of our modelled function should be large enough to correctly map all input domain values.
        return 2 * torch.tanh(0.01 * x)


#######################################################################################
# LIST WISE LOSS FUNCTION MLE
#######################################################################################
"""
Understanding loss_list_wise_mle:
Calculates and returns the probability of the required permutation given 
a list of actual and predicted scores of a set of objects.
E.g: Let the list of scores predicted = [20, 3, 56, 9].
     Let the actual scores = [2.4, 1.3, 3.1, 4.9].
     Given the list of predicted scores, the probability of selecting the given objects
     based on actual scores (required permutation) is to be increased.
     Here this probability is the following:
     Object 4 has the best actual score. It need to be ranked first. Hence the expected ranked list:
        object4 --> (Has actual score 4.9)
        object3 --> (Has actual score 3.1)
        object1 --> (Has actual score 2.4)
        object2 --> (Has actual score 1.3)
     The probability of selecting this list using the predicted score is (without replacement):
     (9/88) * (56/79) * (20/23) * (3/3)
     Applying log to this gives
     log(9/88) + log(56/79) + log(20/23) + log(3/3)
     Since the scores can be negative, we use a strictly increasing positive function like exp
     as a helper and convert all the predictions to exp(predictions)
@Interface:
    predicted_scores: Score list predicted by a score function. Shape expected = [..., listsize]
    actual_scores: Score list known to us. Shape expected = [..., listsize]
@Return:
    Negative log likelihood of required permutation being selected using the predicted score list.
"""


def loss_list_wise_mle(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    # Making sure that the tensor sizes are the same.
    if predicted_scores.size() != actual_scores.size():
        raise Exception("loss_list_wise_mle: predicted_scores.size() != actual_scores.size()")

    # Numerical stability:
    #     Subtracting a constant from predicted_scores has no impact on result.
    #     This is because we use exp as our strictly positive increasing function
    #     to make sure that the calculated probabilities are positive.
    max_predicted, _ = torch.max(predicted_scores, dim=-1, keepdim=True)
    predicted_scores = predicted_scores - max_predicted
    predicted_scores = torch.exp(predicted_scores)

    # Making sure that the given actual score list is unchanged.
    # We need to change the predicted score list for the autograd functionality to work.
    actual_scores = actual_scores.clone()

    # Loop through the object selection process (without replacement) and sum results
    log_probability_sum = 0
    n = actual_scores.shape[-1]  # Number of elements in a list / list size.
    for _ in range(n):
        # Calculate the top 1 log probability and remove the items from both
        # actual and predicted list (Hence the cloning was important)
        log_prob = top_1_log_prob(predicted_scores, actual_scores)
        predicted_scores, actual_scores = remove_top_1_element(predicted_scores, actual_scores)
        log_probability_sum += log_prob

    return -1 * log_probability_sum


def top_1_log_prob(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    # Viewing everything as a 2D tensor.
    y_actual = actual_scores.view(-1, actual_scores.shape[-1])
    y_predicted = predicted_scores.view(-1, predicted_scores.shape[-1])

    # Calculate the max indices and values in the 2D view.
    max_indices = torch.argmax(y_actual, dim=-1)
    max_values = y_predicted[np.arange(max_indices.shape[0]), max_indices]

    # View the max values in (n-1)D view.
    max_values = max_values.view(predicted_scores.shape[:-1])
    prob = max_values / torch.sum(predicted_scores, dim=-1)

    return torch.log(prob)


def remove_top_1_element(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    # Calculate the max indices in the 2D view.
    y_actual = actual_scores.view(-1, actual_scores.shape[-1])
    max_indices = torch.argmax(y_actual, dim=-1)

    # Get the nD max indices however viewing it as 1D. (Note: torch.argmax reduces the dimension by 1)
    max_indices = np.arange(max_indices.shape[0]) * predicted_scores.shape[-1] + max_indices.numpy()
    max_indices = torch.from_numpy(max_indices)

    # View everything with a 1D view.
    y_predicted = predicted_scores.view(-1)
    y_actual = actual_scores.view(-1)

    # Create the mask indices in 1D view and calculate the new shape.
    mask = torch.ones_like(y_predicted, dtype=torch.bool)
    mask[max_indices] = False
    new_view_shape = actual_scores.shape[:-1] + (actual_scores.shape[-1]-1,)

    # Return the remaining elements in the new shaped view.
    return y_predicted[mask].view(new_view_shape), y_actual[mask].view(new_view_shape)


def is_sorted(l, reverse=False):
    # https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not
    if reverse:
        # check if descent sorted
        return all(l[:-1] >= l[1:])
    else:
        # check if ascendent sorted
        return all(l[:-1] <= l[1:])

def test_toy_problem():
    # IDEA:
    #   Can our loss function train the model to sort numbers?
    #   List of numbers to train with = {1 to 100}
    #   List of numbers to validate with = {1 to 100}
    #       Result -> Sorting possible provided the output range of the model not restricted
    #       For now we sample train/val data from the same distribution.
    #   List of numbers to test with = {-1 to -100}
    #       Result -> This is not possible because the model will map this to extreme values as it
    #                 is not distribution dependent.
    #   Check the percentage of the lists in the correct sorted order.
    epochs = 1000
    sc = Scorer(input_dim=1)

    # Creating a uniform distribution between [r1, r2]
    r1 = 0
    r2 = 100
    get_training_data = lambda x: (r1 - r2) * torch.rand(x + (1,)) + r2

    optimizer = torch.optim.Adam(sc.parameters(), lr=0.0001)
    for _ in range(epochs):

        optimizer.zero_grad()

        # Changing the size for it to pass through our scorer
        train_data = get_training_data((10, 10, 3))
        prediction = sc(train_data)

        flatten_from_dim = len(train_data.shape) - 2
        prediction = torch.flatten(prediction, start_dim=flatten_from_dim)
        train_data = torch.flatten(train_data, start_dim=flatten_from_dim)

        loss = loss_list_wise_mle(prediction, -1 * train_data)
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()

        print("Epoch[", _, "] ==>", loss.item())

    print("Now testing")
    sorted_lists = 0
    for i in range(1000):

        # Changing the size for it to pass through our scorer
        train_data = get_training_data((100,))
        prediction = sc(train_data)

        prediction = prediction.flatten()
        train_data = train_data.flatten()

        sorted_indices = torch.argsort(prediction)
        train_data_predicted_rank = train_data[sorted_indices].numpy()

        if is_sorted(train_data_predicted_rank, reverse=True):
            sorted_lists += 1

    print("Sorted percentage : ", sorted_lists * 100 / 1000)


def get_batch_HPBO(meta_data, batch_size, list_size, random_state=None):
    support_X = []
    support_y = []
    query_X = []
    query_y = []

    rand_num_gen = np.random.RandomState(seed=random_state)

    # Sample all tasks and form a high dimensional tensor of size
    #   (tasks, batch_size, list_size, input_dim)
    #   Suggestion : Take a tensor batch_size, list_size, input_dim for one gradient step.
    #   For segregation. https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html
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

def get_batch_HPBO_single(meta_train_data, batch_size, list_size):
    support_size = 20  # 5 + np.random.choice(95)  # With 20 it was a good result curve
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


class RankingLossSurrogate(nn.Module):
    def __init__(self, input_dim, file_name=None):
        super(RankingLossSurrogate, self).__init__()
        self.save_folder = "./save/rlsurrogates_deepset_16_5000_uncertainty/"
        self.file_name = file_name
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        if file_name:
            self.load(file_name)
        else:
            self.input_dim = input_dim
            self.ds_embedder, self.sc = self.create_embedder_scorers_uncertainty(self.input_dim)

    def create_embedder_scorer(self, in_dim):
        embedder = DeepSet(input_dim=in_dim+1, latent_dim=32, output_dim=16)
        sc = nn.ModuleList([Scorer(input_dim=16 + in_dim)])
        return embedder, sc

    def create_embedder_scorers_uncertainty(self, in_dim):
        embedder = DeepSet(input_dim=in_dim+1, latent_dim=32, output_dim=16)
        sc_list = []
        for i in range(5):
            sc_list += [Scorer(input_dim=16 + in_dim)]
        return embedder, nn.ModuleList(sc_list)

    def save(self, file_name):
        file_name = self.save_folder + file_name
        state_dict = self.sc.state_dict()
        embedder_state_dict = self.ds_embedder.state_dict()
        torch.save({"input_dim": self.input_dim,
                    "scorer": state_dict,
                    "deep_set": embedder_state_dict},
                   file_name)

    def load(self, file_name):
        file_name = self.save_folder + file_name
        state_dict = torch.load(file_name)
        self.input_dim = state_dict["input_dim"]
        # Creating and initializing the deep set embedder and scorer
        self.ds_embedder, self.sc = self.create_embedder_scorers_uncertainty(self.input_dim)
        self.ds_embedder.load_state_dict(state_dict["deep_set"])
        self.sc.load_state_dict(state_dict["scorer"])

    def forward_mean_var(self, input):
        s_X, s_y, q_X = input

        # Creating an embedding of X:y for the support data using the embedder
        s_X = torch.cat((s_X, s_y), dim=-1)
        s_X = self.ds_embedder(s_X)

        # Creating an input for the scorer.
        s_X = s_X[..., None, :]
        repeat_tuple = (1,) * (len(s_X.shape) - 2) + (q_X.shape[-2], 1)
        s_X = s_X.repeat(repeat_tuple)
        q_X = torch.cat((s_X, q_X), dim=-1)

        # return self.sc(q_X)
        predictions = []
        for s in self.sc:
            predictions += [s(q_X)]

        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0, keepdim=True)[0]

        return mean, variance

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

        # return self.sc(q_X)
        predictions = []
        for s in self.sc:
            predictions += [s(q_X)]

        # return predictions[0]

        predictions = torch.stack(predictions)
        return torch.mean(predictions, dim=0)  #  self.sc(q_X)
        # return torch.sum(predictions, dim=0)  # self.sc(q_X)

    def flatten_for_loss_list(self, pred, y):
        flatten_from_dim = len(pred.shape) - 2
        pred = torch.flatten(pred, start_dim=flatten_from_dim)
        y = torch.flatten(y, start_dim=flatten_from_dim)
        return pred, y


    def train_model2(self, get_batch, epochs, list_size):
            optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': 0.001},])  # 0.0001 giving good results
            loss_list = []
            for _ in range(epochs):
                self.train()
                # optimizer.zero_grad()

                batch_loss = []
                for __ in range(100):
                    optimizer.zero_grad()
                    s_train_X, s_train_y, q_train_X, q_train_y = get_batch(list_size)

                    prediction = self.forward((s_train_X, s_train_y, q_train_X))
                    prediction, q_train_y = self.flatten_for_loss_list(prediction, q_train_y)

                    # Viewing everything as a 2D tensor.
                    q_train_y = q_train_y.view(-1, q_train_y.shape[-1])
                    prediction = prediction.view(-1, prediction.shape[-1])

                    loss = listMLE(prediction, q_train_y)
                    # loss = loss_list_wise_mle(prediction, q_train_y)
                    # loss = torch.mean(loss)

                    loss.backward()
                    optimizer.step()
                    batch_loss += [loss]


                    val_loss = np.array([np.nan])

                print("Epoch[", _, "] ==> Loss =", loss.item() / list_size, "; Val_loss =", val_loss.item() / list_size)
                loss_list += [loss.item() / list_size]


            return loss_list, None

    def get_fine_tune_batch(self, X_obs, y_obs):

        # Taking 20% of the data as the support set.
        support_size = int(0.2 * X_obs.shape[0])
        idx_support = np.random.choice(X_obs.shape[0], size=support_size, replace=False)
        idx_query = np.delete(np.arange(X_obs.shape[0]), idx_support)

        s_ft_X = X_obs[idx_support]
        s_ft_y = y_obs[idx_support]
        q_ft_X = X_obs[idx_query]
        q_ft_y = y_obs[idx_query]

        return s_ft_X, s_ft_y, q_ft_X, q_ft_y

    def train_model(self, meta_train_data, meta_val_data, epochs, batch_size, list_size, search_space_id):
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': 0.001},])  # 0.0001 giving good results
        loss_list = []
        val_loss_list = []
        for _ in range(epochs):
            self.train()
            # optimizer.zero_grad()

            batch_loss = []
            for __ in range(100):
                optimizer.zero_grad()
                s_train_X, s_train_y, q_train_X, q_train_y = get_batch_HPBO_single(meta_train_data, 1, list_size)

                prediction = self.forward((s_train_X, s_train_y, q_train_X))
                prediction, q_train_y = self.flatten_for_loss_list(prediction, q_train_y)

                # Viewing everything as a 2D tensor.
                q_train_y = q_train_y.view(-1, q_train_y.shape[-1])
                prediction = prediction.view(-1, prediction.shape[-1])

                loss = listMLE(prediction, q_train_y)
                # loss = loss_list_wise_mle(prediction, q_train_y)
                # loss = torch.mean(loss)

                loss.backward()
                optimizer.step()
                batch_loss += [loss]

            with torch.no_grad():
                self.eval()
                # Calculating training loss
                s_train_X, s_train_y, q_train_X, q_train_y = get_batch_HPBO(meta_train_data, batch_size, list_size)

                prediction = self.forward((s_train_X, s_train_y, q_train_X))
                prediction, q_train_y = self.flatten_for_loss_list(prediction, q_train_y)

                # Viewing everything as a 2D tensor.
                q_train_y = q_train_y.view(-1, q_train_y.shape[-1])
                prediction = prediction.view(-1, prediction.shape[-1])

                loss = listMLE(prediction, q_train_y)

                # Calculating validation loss
                s_val_X, s_val_y, q_val_X, q_val_y = get_batch_HPBO(meta_val_data, batch_size, list_size, int(search_space_id))

                pred_val = self.forward((s_val_X, s_val_y, q_val_X))
                pred_val, q_val_y = self.flatten_for_loss_list(pred_val, q_val_y)

                # Viewing everything as a 2D tensor.
                q_val_y = q_val_y.view(-1, q_val_y.shape[-1])
                pred_val = pred_val.view(-1, pred_val.shape[-1])

                val_loss = listMLE(pred_val, q_val_y)
                # val_loss = torch.mean(val_loss)

            # if val_loss.item() < min(val_loss_list + [np.inf]):
            #    self.save(search_space_id)

            print("Epoch[", _, "] ==> Loss =", loss.item() / list_size, "; Val_loss =", val_loss.item() / list_size)
            loss_list += [loss.item() / list_size]
            val_loss_list += [val_loss.item() / list_size]

        self.save(search_space_id)

        return loss_list, val_loss_list

    def get_fine_tune_batch(self, X_obs, y_obs):

        # Taking 20% of the data as the support set.
        support_size = int(0.2 * X_obs.shape[0])
        idx_support = np.random.choice(X_obs.shape[0], size=support_size, replace=False)
        idx_query = np.delete(np.arange(X_obs.shape[0]), idx_support)

        s_ft_X = X_obs[idx_support]
        s_ft_y = y_obs[idx_support]
        q_ft_X = X_obs[idx_query]
        q_ft_y = y_obs[idx_query]

        return s_ft_X, s_ft_y, q_ft_X, q_ft_y

    def fine_tune(self, X_obs, y_obs):
        epochs = 300
        loss_list = []
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': 0.001},])
        scheduler_fn = lambda x, y: torch.optim.lr_scheduler.CosineAnnealingLR(x, y, eta_min=0.0001)
        scheduler = scheduler_fn(optimizer, epochs)
        for i in range(epochs):
            self.train()
            optimizer.zero_grad()

            s_ft_X, s_ft_y, q_ft_X, q_ft_y = self.get_fine_tune_batch(X_obs, y_obs)

            q_ft_y_pred = self.forward((s_ft_X, s_ft_y, q_ft_X))
            q_ft_y_pred, q_ft_y = self.flatten_for_loss_list(q_ft_y_pred, q_ft_y)

            q_ft_y = q_ft_y.view(-1, q_ft_y.shape[-1])
            q_ft_y_pred = q_ft_y_pred.view(-1, q_ft_y_pred.shape[-1])

            loss = listMLE(q_ft_y_pred, q_ft_y)

            # loss = loss_list_wise_mle(q_ft_y_pred, q_ft_y) # !!!!! NOt using loss ... issues..
            # loss = torch.mean(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # if loss.item() < min(loss_list + [float('inf')]):
            #    self.save(self.file_name + "_ft_early_stop")

            loss_list += [loss.item()/X_obs.shape[0]]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss: Ranking losses"]
        plt.legend(legend)
        plt.title("SSID: " + self.file_name + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.file_name + "_fine_tune_loss.png")
        plt.close()

        # self.save(self.file_name + "_ft_early_stop")
        # self.load(self.file_name + "_ft_early_stop")

        return loss_list

    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        X_obs = torch.from_numpy(X_obs)
        y_obs = torch.from_numpy(y_obs)
        X_pen = torch.from_numpy(X_pen)

        # Doing restarts from the saved model
        restarted_model = RankingLossSurrogate(input_dim=-1, file_name=self.file_name)
        restarted_model.fine_tune(X_obs, y_obs) # disabling fine tuning for now
        scores = acquisition_EI(X_obs, y_obs, X_pen, restarted_model)
        # restarted_model((X_obs, y_obs, X_pen))
        # scores = scores.detach().numpy()

        idx = np.argmax(scores)
        return idx

def pre_train_HPOB():
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3")

    # Pretrain Ranking loss surrogate with all search spaces
    for search_space_id in hpob_hdlr.get_search_spaces():
        meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
        meta_val_data = hpob_hdlr.meta_validation_data[search_space_id]

        input_dim = get_input_dim(meta_train_data)
        print("Input dim of", search_space_id, "=", input_dim)

        meta_train_data = convert_meta_data_to_np_dictionary(meta_train_data)
        meta_val_data = convert_meta_data_to_np_dictionary(meta_val_data)

        epochs = 5000
        batch_size = 100
        list_size = 100
        rlsurrogate = RankingLossSurrogate(input_dim=input_dim)
        loss_list, val_loss_list = \
            rlsurrogate.train_model(meta_train_data, meta_val_data, epochs, batch_size, list_size, search_space_id)

        rlsurrogate.load(search_space_id)

        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        plt.plot(np.array(val_loss_list, dtype=np.float32))
        legend = ["Loss",
                  "Validation Loss"
                  ]
        plt.legend(legend)
        plt.title("SSID: " + search_space_id + "; Input dim: " + str(input_dim))
        plt.savefig(rlsurrogate.save_folder + "loss_" + search_space_id + ".png")

if __name__ == '__main__':
    # Unit testing our loss functions
    # test_toy_problem()

    pre_train_HPOB()
