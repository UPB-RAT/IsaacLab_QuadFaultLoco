# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

class TinyPINN(nn.Module):
    def __init__(
            self, 
            num_prop, 
            num_hist,
            num_latent,
            activation="elu",
            encoder_net_type="gru"
            ):
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.num_prop = num_prop
        self.num_latent = num_latent
        self.encoder_net_type = encoder_net_type 

        if encoder_net_type == "gru":
            layers = [
                      nn.Linear(num_prop, num_latent),
                      activation,
                      nn.GRU(num_latent, num_latent, num_layers=1, batch_first=True),
                     ]
            # layers = [
            #           nn.Conv1d(num_hist, 32),
            #           activation,
            #           nn.GRU(num_latent, num_latent, num_layers=1, batch_first=True),
            #          ]
            # layers = [
            #           nn.Linear(num_prop, 128),
            #           activation,
            #           nn.Linear(128, num_latent),
            #           activation,
            #           nn.GRU(num_latent, num_latent, num_layers=1, batch_first=True),
            #          ]
        elif encoder_net_type == "cnn":
            layers = [
                        nn.Conv1d(num_hist,32,9,2),
                        activation,
                        nn.Conv1d(32,32,5,1),
                        activation,
                        nn.Conv1d(32,32,5,1),
                        activation,
                        nn.Flatten(),
                     ]
        elif encoder_net_type == "mlp":
            layers = [
                        nn.Flatten(),
                        nn.Linear(num_prop * num_hist, 512),
                        activation,
                        nn.Linear(512, 256),
                        activation,
                        nn.Linear(256, 128),
                        activation,
                     ]
        if encoder_net_type == "mlp" or encoder_net_type == "cnn":
            last_layer = nn.LazyLinear(num_latent)
            layers.append(last_layer)
        self.encoder = nn.Sequential(*layers)
        # self.encode_mean_latent = nn.Conv1d(num_hist,1,1,1)
        # self.encode_logvar_latent = nn.Conv1d(num_hist,1,1,1)
     
        # # heads
        self.Minv_head = nn.Linear(num_latent, 78)   # next state
        self.fn_head = nn.Linear(num_latent, 12)   # next state
        # self.dq_head = nn.Linear(num_latent, 12)   # next state
        self.fault_head = nn.Linear(num_latent, 12)
        # heads
        # self.Minv_head = nn.Sequential(nn.Linear(num_latent, 128),
        #                                activation,
        #                                nn.Linear(128, 78)) 
        # self.fn_head = nn.Sequential(nn.Linear(num_latent, 64),
        #                                activation,
        #                                nn.Linear(64, 12)) 
        # # self.dq_head = nn.Linear(num_latent, 12) 
        # self.fault_head = nn.Sequential(nn.Linear(num_latent, 32),
        #                                activation,
        #                                nn.Linear(32, 13)) 

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def latent_forward(self, obs_hist):
        # breakpoint()
        if self.encoder_net_type == "gru":
            o_n, h_n = self.encoder(obs_hist)
            return o_n.mean(1)
        else:
            return self.encoder(obs_hist)

        # mean_latent = self.encode_mean_latent(o_n)
        # logvar_latent = self.encode_logvar_latent(o_n)
        # code_latent = self.reparameterise(mean_latent,logvar_latent)
        # return code_latent.squeeze(1), mean_latent, logvar_latent

        # return self.conv_encoder(obs_hist)
    
    def predict(self, obs_hist, priv_obs, dt = None):

        latent = self.latent_forward(obs_hist)
        # latent, mean_latent, logvar_latent = self.latent_forward(obs_hist)
        # breakpoint()
        # M inverse
        Lterms = self.Minv_head(latent)
        relu = resolve_nn_activation("relu")
        L = torch.zeros((Lterms.shape[0], 12, 12), device=obs_hist.device)
        idx = torch.tril_indices(12, 12, offset=-1)
        L[:, idx[0], idx[1]] = Lterms[:, 12:]
        L += torch.diag_embed(relu(Lterms[:, :12]) + 0.1)
        Minv = L @ L.transpose(1,2)
        # fn and delta q
        fn = self.fn_head(latent)
        # dq = self.dq_head(latent)
        
        # integrate
        # breakpoint()
        qddot = Minv @ (priv_obs[:,-12:] + fn).unsqueeze(-1)
        if dt is None:
            dt = 0.005
        pred_qdot = obs_hist[:,-1,12:24] + qddot.squeeze(-1) * dt 
        # pred_q    = obs_hist[:,-1,:12] + obs_hist[:,-1,12:24] * dt + dq
        # breakpoint()
        pred_q    = obs_hist[:,-1,:12] + pred_qdot * dt

        fault = self.fault_head(latent)

        return pred_q, pred_qdot, fault
        # return pred_q, pred_qdot.squeeze(-1), fault, mean_latent, logvar_latent
    
class ActorCriticPINN(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_prop,
        num_hist,
        num_critic_obs,
        num_actions,
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
        num_latent = 32
        mlp_input_dim_a = self.num_prop + num_latent
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

        self.pinn = TinyPINN(num_prop,
                             num_hist=num_hist,
                             num_latent=32)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Adaptation Module PINN: {self.pinn}")

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
    
    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        mean = torch.clamp(mean, -5.0, 5.0)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        std = torch.clamp(std, 0.01, 1.0)
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, obs_hist, **kwargs):
        # breakpoint()
        latent = self.pinn.latent_forward(obs_hist)
        # latent, _, _ = self.pinn.latent_forward(obs_hist)
        obs = torch.cat((obs, latent),dim=-1)
        self.update_distribution(obs)
        try:
            action = self.distribution.sample()
        except:
            breakpoint()
        return action

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, obs_hist):
        latent = self.pinn.latent_forward(obs_hist)
        # latent, _, _ = self.pinn.latent_forward(obs_hist)
        obs = torch.cat((obs, latent),dim=-1)
        actions_mean = self.actor(obs)
        return actions_mean, latent

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

