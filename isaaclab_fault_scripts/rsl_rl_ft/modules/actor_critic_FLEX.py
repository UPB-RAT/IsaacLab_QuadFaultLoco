# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticFLEX(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_prop,
        num_hist,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
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
        self.latent_dim = 16
        self.num_prop = num_prop
        mlp_input_dim_a = self.num_prop + self.latent_dim + 3 + 12
        mlp_input_dim_c = num_critic_obs

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

        self.encoder = nn.Sequential(
            nn.Linear(num_prop*num_hist,512),
            activation,
            nn.Linear(512,256),
            activation,
            nn.Linear(256,128),
            activation,
        )
        self.encode_mean_latent = nn.Linear(128,self.latent_dim)
        self.encode_logvar_latent = nn.Linear(128,self.latent_dim)
        self.encode_mean_vel = nn.Linear(128,3)
        self.encode_logvar_vel = nn.Linear(128,3)
        self.encode_fault_logit = nn.Linear(128,12)

        self.fault_modulation_head = nn.Sequential(
            nn.Linear(12,64),
            activation,
            nn.Linear(64,64),
            activation,
            nn.Linear(64,self.latent_dim * 2),
            activation,
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim,64),
            activation,
            nn.Linear(64,128),
            activation,
            nn.Linear(128,self.num_prop)
        )
        
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

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def femnet_forward(self, obs_history):
        if len(obs_history.shape) > 2:
            obs_history = obs_history.view(obs_history.shape[0], -1)
        distribution = self.encoder(obs_history)

        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)

        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)

        code_latent = self.reparameterise(mean_latent,logvar_latent)
        code_vel = self.reparameterise(mean_vel,logvar_vel)

        
        fault_logit = self.encode_fault_logit(distribution)
        fault_label = torch.max(fault_logit, dim = -1)[1]
        fault_binarylabel = torch.zeros_like(fault_logit).to(fault_logit.device)
        fault_binarylabel[torch.arange(fault_binarylabel.shape[0]), fault_label] = 1.

        gamma = self.fault_modulation_head(fault_binarylabel)
        code_latent = code_latent * gamma[:,:self.latent_dim] + gamma[:,self.latent_dim:]

        decode = self.decoder(code_latent)

        code = torch.cat((code_vel, fault_binarylabel, code_latent),dim=-1)

        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent, fault_logit

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

    def act(self, obs, obs_history, **kwargs):
        # breakpoint()
        code,_,_,_,_,_,_,_ = self.femnet_forward(obs_history)
        obs = torch.cat((code, obs),dim=-1)
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, obs_history):
        code,_,_,_,_,_,_,_ = self.femnet_forward(obs_history)
        obs = torch.cat((code, obs),dim=-1)
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
