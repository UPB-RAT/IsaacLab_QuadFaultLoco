# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torchdiffeq import odeint_adjoint, odeint_event
from rsl_rl.utils import resolve_nn_activation

class GRUEncoder(nn.Module):
    def __init__(self, ninput=49, nhidden=64, nlayer=1, nout=16):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_size=ninput,
                          hidden_size=nhidden, 
                          num_layers=nlayer,
                          batch_first=True)
        self.proj = nn.LazyLinear(nout)

    def forward(self, x):
        output, h_n = self.gru(x)
        x = self.proj(output)
        return x

class LatentODEfunc(nn.Module):
    def __init__(self, latent_size=16, nhidden=64, activation="elu"):
        super(LatentODEfunc, self).__init__()
        self.activation = resolve_nn_activation(activation)
        self.fc1 = nn.Linear(latent_size, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_size)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out
    
class ActorCriticFLEX(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_prop,
        num_hist,
        num_critic_obs,
        num_actions,
        latent_size = 16,
        sampling_rate = 1/60,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.num_prop = num_prop
        mlp_input_dim_a = self.num_prop + 12
        mlp_input_dim_c = num_critic_obs
        self.latent_size = latent_size
        self.num_hist = num_hist
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.gru_encoder = GRUEncoder(num_prop, 64, 1, latent_size*2)
        
        self.latent_FLEX_net = LatentODEfunc(latent_size, nhidden=64)

        self.cnn_decoder = nn.Sequential(nn.Conv1d(30,32,7,2),
                                         activation,
                                         nn.Conv1d(32,32,5,1),
                                         activation,
                                         nn.Flatten(),
                                         nn.LazyLinear(12))
        
        self.mlp_decoder = nn.Sequential(nn.Linear(latent_size,128),
                                            activation,
                                            nn.Linear(128,num_prop)
                                            )
        self.sampling_rate = sampling_rate
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def encoder_forward(self, obs_hist):
        # breakpoint()
        for t in reversed(range(obs_hist.size(1))):
            obs = obs_hist[:, t, :]
            out = self.gru_encoder.forward(obs)
        # breakpoint()
        qz0_mean, qz0_logvar = out[:, :self.latent_size], out[:, self.latent_size:]
        epsilon = torch.randn(qz0_mean.size()).to(qz0_mean.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean # latent of initial point
        return z0, qz0_mean, qz0_logvar
    
    def latent_FLEX_forward(self, z0):
        # breakpoint()
        samp_time = torch.arange(0, self.num_hist*self.sampling_rate, self.sampling_rate).to(z0.device)
        pred_latent = odeint_adjoint(self.latent_FLEX_net, z0, samp_time).permute(1, 0, 2)
        return pred_latent
    
    def decoder_forward(self, pred_latent):
        # breakpoint()
        pred_fault = self.cnn_decoder(pred_latent)
        pred_trajectory = self.mlp_decoder(pred_latent)
        return pred_fault, pred_trajectory
    
    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        if (std < 1e-5).sum() > 0:
            breakpoint()
            std[:] = 1e-2
        self.distribution = Normal(mean, std)

    def act(self, obs, obs_hist, **kwargs):
        # breakpoint()
        z0, _, _ = self.encoder_forward(obs_hist)
        pred_z = self.latent_FLEX_forward(z0)
        # pred_obs_hist = self.decoder_forward(pred_z)
        pred_fault = self.cnn_decoder(pred_z)
        obs = torch.cat((obs, pred_fault),dim=-1)
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, obs_hist):
        z0 = self.encoder_forward(obs_hist)
        pred_z = self.latent_FLEX_forward(z0)
        obs = torch.cat((obs, pred_z),dim=-1)
        actions_mean = self.actor(obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
