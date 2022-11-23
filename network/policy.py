import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp, reparameterize, evaluate_lop_pi

from torch.distributions.categorical import Categorical

class decrete_Policy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()
        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
    
    def forward(self,states):
        probs = F.softmax(self.net(states),dim=1)
        return torch.tensor([torch.argmax(probs)])
    
    def sample(self,states):
        probs = F.softmax(self.net(states),dim=1)
        dis_ = Categorical(probs=probs)
        actions = dis_.sample()
        return actions, dis_.log_prob(actions)
    
    def evaluate_log_pi(self, states, actions):
        probs = F.softmax(self.net(states),dim=1)
        return torch.log(probs.gather(1, actions.type(torch.int64)))

class StateIndependentPolicy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))
