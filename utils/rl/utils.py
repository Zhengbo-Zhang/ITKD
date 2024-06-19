import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(object):
    def __init__(
        self,
        state_dim,
        discrete_action_dim,
        parameter_action_dim,
        all_parameter_action_dim,
        discrete_emb_dim,
        parameter_emb_dim,
        max_size=int(1e6),
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.parameter_action = np.zeros((max_size, parameter_action_dim))
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim))

        self.discrete_emb = np.zeros((max_size, discrete_emb_dim))
        self.parameter_emb = np.zeros((max_size, parameter_emb_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.state_next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state,
        discrete_action,
        parameter_action,
        all_parameter_action,
        discrete_emb,
        parameter_emb,
        next_state,
        state_next_state,
        reward,
        done,
    ):
        self.state[self.ptr] = state
        self.discrete_action[self.ptr] = discrete_action
        self.parameter_action[self.ptr] = parameter_action
        self.all_parameter_action[self.ptr] = all_parameter_action
        self.discrete_emb[self.ptr] = discrete_emb
        self.parameter_emb[self.ptr] = parameter_emb
        self.next_state[self.ptr] = next_state
        self.state_next_state[self.ptr] = state_next_state

        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.discrete_action[ind]).to(self.device),
            torch.FloatTensor(self.parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.all_parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.discrete_emb[ind]).to(self.device),
            torch.FloatTensor(self.parameter_emb[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.state_next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


class RewardRedistribution(nn.Module):
    def __init__(self, n_positions, n_actions, n_lstm) -> None:
        super().__init__()
        def identity(x):
            return x
        
        
        # self.lstm = torch.nn.LSTM(
        #     input_size=n_positions+n_actions,
        #     hidden_size=n_lstm,
        # )
        
        self.lstm = nn.Sequential(
            nn.Linear(n_positions+n_actions, n_lstm),
            nn.Identity(),
        )
        
        # self.lstm = LSTMLayer(
        #     in_features=n_positions + n_actions,
        #     out_features=n_lstm,
        #     inputformat="NLC",
        #     # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
        #     w_ci=(torch.nn.init.xavier_normal_, False),
        #     # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
        #     w_ig=(False, torch.nn.init.xavier_normal_),
        #     # output gate: disable all connection (=no forget gate) and disable bias
        #     w_og=False,
        #     b_og=False,
        #     # forget gate: disable all connection (=no forget gate) and disable bias
        #     w_fg=False,
        #     b_fg=False,
        #     # LSTM output activation is set to identity function
        #     a_out=identity,
        # )

        # After the LSTM layer, we add a fully connected output layer
        self.fc_out = torch.nn.Linear(n_lstm, 1)

    def forward(self, observations, actions):
        # Process input sequence by LSTM
        # lstm_out, _ = self.lstm(
        #     torch.cat([observations, actions], dim=-1),
        #     return_all_seq_pos=True,  # return predictions for all sequence positions
        # )
        lstm_out = self.lstm(torch.cat([observations, actions], dim=-1))
        net_out = self.fc_out(lstm_out)
        return net_out


class CriticalStateDetector(nn.Module):
    def __init__(self, n_positions, n_lstm) -> None:
        super().__init__()
        
        # self.lstm = torch.nn.LSTM(
        #     n_positions,
        #     n_lstm,
        # )
        
        self.lstm = nn.Sequential(
            nn.Linear(n_positions, n_lstm),
            nn.Identity(),
        )
        
        # self.lstm = LSTMLayer(
        #     in_features=n_positions,
        #     out_features=n_lstm,
        #     inputformat="NLC",
        #     # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
        #     w_ci=(torch.nn.init.xavier_normal_, False),
        #     # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
        #     w_ig=(False, torch.nn.init.xavier_normal_),
        #     # output gate: disable all connection (=no forget gate) and disable bias
        #     w_og=False,
        #     b_og=False,
        #     # forget gate: disable all connection (=no forget gate) and disable bias
        #     w_fg=False,
        #     b_fg=False,
        #     # LSTM output activation is set to identity function
        #     a_out=lambda x: x,
        # )

        # After the LSTM layer, we add a fully connected output layer
        self.fc_out = torch.nn.Linear(n_lstm, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, observations):
        # Process input sequence by LSTM
        # lstm_out, _ = self.lstm(
        #     observations,
        #     return_all_seq_pos=True,  # return predictions for all sequence positions
        # )
        lstm_out = self.lstm(observations)
        net_out = self.sigmoid(self.fc_out(lstm_out))
        return net_out


def lossfunction_rew(predictions, returns):
    """
    rewards.shape = (batch_size, n_actions)
    rewards = [0,0,0 ...,前batch个图片精度]
    """
    # Main task: predicting return at last timestep
    main_loss = torch.mean(predictions[:, -1] - returns) ** 2
    # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
    aux_loss = torch.mean(predictions[:, :] - returns.unsqueeze(-1)) ** 2
    # Combine losses
    loss = main_loss + aux_loss * 0.5
    return loss
