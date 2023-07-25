import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.ader import ADERMixer, ADERMixer_Ent
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop, Adam

import numpy as np
from torch.distributions import Categorical
from modules.critics.ader import ADERCritic
from utils.rl_utils import build_td_lambda_targets
import pdb, math
from torch.autograd import Variable

alpha_min = 1e-4
alpha_max =  0.5

class ADER_Leaner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic1 = ADERCritic(scheme, args)
        self.critic2 = ADERCritic(scheme, args)
        self.mixer1 = ADERMixer(args)
        self.mixer2 = ADERMixer(args)
        self.target_mixer1 = copy.deepcopy(self.mixer1)
        self.target_mixer2 = copy.deepcopy(self.mixer2)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic1_ent = ADERCritic(scheme, args)
        self.critic2_ent = ADERCritic(scheme, args)
        self.mixer1_ent = ADERMixer_Ent(args)
        self.mixer2_ent = ADERMixer_Ent(args)
        self.target_mixer1_ent = copy.deepcopy(self.mixer1_ent)
        self.target_mixer2_ent = copy.deepcopy(self.mixer2_ent)
        self.target_critic1_ent = copy.deepcopy(self.critic1_ent)
        self.target_critic2_ent = copy.deepcopy(self.critic2_ent)

        
        self.agent_params = list(mac.parameters())
        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters())
        self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())
        self.critic_ent_params1 = list(self.critic1_ent.parameters()) + list(self.mixer1_ent.parameters())
        self.critic_ent_params2 = list(self.critic2_ent.parameters()) + list(self.mixer2_ent.parameters())


        if self.args.optimizer == "Adam":
            self.p_optimiser = Adam(params=self.agent_params, lr=args.lr)
            self.c_optimiser1 = Adam(params=self.critic_params1, lr=args.c_lr)
            self.c_optimiser2 = Adam(params=self.critic_params2, lr=args.c_lr)
            self.c_ent_optimiser1 = Adam(params=self.critic_ent_params1, lr=args.c_lr)
            self.c_ent_optimiser2 = Adam(params=self.critic_ent_params2, lr=args.c_lr)
        else:
            self.p_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.c_optimiser1 = RMSprop(params=self.critic_params1, lr=args.c_lr, alpha=args.optim_alpha,
                                        eps=args.optim_eps)
            self.c_optimiser2 = RMSprop(params=self.critic_params2, lr=args.c_lr, alpha=args.optim_alpha,
                                        eps=args.optim_eps)
            self.c_ent_optimiser1 = RMSprop(params=self.critic_ent_params1, lr=args.c_lr, alpha=args.optim_alpha,
            eps = args.optim_eps)
            self.c_ent_optimiser2 = RMSprop(params=self.critic_ent_params2, lr=args.c_lr, alpha=args.optim_alpha,
            eps = args.optim_eps)

        self.log_alpha = Variable(self.args.adap_total_alpha_start * th.ones(self.n_agents).cuda(),
                                  requires_grad=True)  # .cuda()
        self.alpha = th.exp(self.log_alpha)
        self.maximum_entropy_sum = np.ones(self.n_agents) * np.log(self.n_actions)
        self.target_entropy_sum = self.maximum_entropy_sum * self.args.tar_ent_ratio
        self.alpha_params = [self.log_alpha]
        if self.args.optimizer == "Adam":
            self.alpha_optimiser = Adam(params=self.alpha_params, lr=args.c_lr)
        else:
            self.alpha_optimiser = RMSprop(params=self.alpha_params, lr=args.c_lr, alpha=args.optim_alpha,
                                           eps=args.optim_eps)

    
    def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int, masked_td_error1_ent:None, masked_td_error2_ent:None):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # mask = mask.repeat(1, 1, self.n_agents).view(-1)
        avail_actions = batch["avail_actions"]
        states = batch["state"]

        mac = self.mac

        alpha = self.alpha.detach()
        alpha = th.clamp(alpha, max=alpha_max, min=alpha_min)

        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 1e-10
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        pi = mac_out.clone()
        log_pi = th.log(pi)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)
        v_vals1_ = th.sum(q_vals1 * mac_out, dim=-1)
        v_vals2_ = th.sum(q_vals2 * mac_out, dim=-1)
        v_vals1 = self.mixer1(v_vals1_, states)
        v_vals2 = self.mixer2(v_vals2_, states)
        v_tot = th.min(v_vals1, v_vals2)

        q_vals1_ent = self.critic1_ent.forward(inputs)
        q_vals2_ent = self.critic2_ent.forward(inputs)
        v_vals1_ent = th.sum(q_vals1_ent * mac_out, dim=-1)
        v_vals2_ent = th.sum(q_vals2_ent * mac_out, dim=-1)
        v_vals1_ent = self.mixer1_ent(v_vals1_ent, states)
        v_vals2_ent = self.mixer2_ent(v_vals2_ent, states)
        entropies = - (pi * log_pi).sum(dim=-1)

        v_tot_min_ind = th.unsqueeze(th.argmin(th.cat([v_vals1, v_vals2], dim=2), dim=-1), dim=-1).repeat(1, 1,
                                                                                                          self.args.n_agents)
        contribution1 = th.autograd.grad((v_vals1[:, :-1] * mask).sum(), v_vals1_, retain_graph=True)[0]
        contribution2 = th.autograd.grad((v_vals2[:, :-1] * mask).sum(), v_vals2_, retain_graph=True)[0]
        contribution_ = v_tot_min_ind * contribution2 + (1 - v_tot_min_ind) * contribution1


        ###
        grad_ent = th.autograd.grad(entropies.sum(), mac_out, retain_graph=True)[0]
        mac_out_2 = mac_out + 1e-5 * grad_ent
        mac_out_2[avail_actions == 0] = 1e-10
        mac_out_2 = mac_out_2 / mac_out_2.sum(dim=-1, keepdim=True)
        mac_out_2[avail_actions == 0] = 1e-10

        v_vals1_2 = th.sum(q_vals1 * mac_out_2, dim=-1)
        v_vals2_2 = th.sum(q_vals2 * mac_out_2, dim=-1)
        entropies2 = - (mac_out_2 * th.log(mac_out_2)).sum(dim=-1)

        entropies_del = entropies2 - entropies
        v_val1_del = v_vals1_2 - v_vals1_
        v_val2_del = v_vals2_2 - v_vals2_

        ratio_vh1 = v_val1_del / (entropies_del + 1e-8)
        ratio_vh2 = v_val2_del / (entropies_del + 1e-8)

        ratio_vh = v_tot_min_ind * ratio_vh1 + (1 - v_tot_min_ind) * ratio_vh2
        contribution = - contribution_ * ratio_vh
        if self.args.contribution_temperature == -1000:
            contribution = contribution / (th.unsqueeze(th.max(contribution, dim=-1)[0], dim=-1) * 0.1 + 1e-5)
            contribution_softmax = th.softmax(-contribution / 1, dim=2)
        else:
            contribution_softmax = th.softmax(-contribution / self.args.contribution_temperature, dim=2)

        weight_v_ent = th.zeros_like(v_vals1_ent[:, :-1])
        weight_v_ent = weight_v_ent + contribution_softmax[:, :-1]

        weight_v_ent_ = (weight_v_ent / 1).detach()
        weight_v_ent_softmax_avg = th.mean(weight_v_ent_, dim=1).mean(dim=0)
        self.target_entropy_sum = (self.args.adap_total_alpha_tau) * self.target_entropy_sum + (
                    1 - self.args.adap_total_alpha_tau) * weight_v_ent_softmax_avg.cpu().detach().numpy() * sum(
            self.target_entropy_sum)

        weight_v_ent = alpha
        weighted_v_ent_tot =((th.min(v_vals1_ent[:, :-1], v_vals2_ent[:, :-1])*weight_v_ent).sum(dim=-1)) + (entropies[:, :-1] * weight_v_ent).sum(dim=-1)

        # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
        pol_target = v_tot[:, :-1, 0] + weighted_v_ent_tot
        policy_loss = -(pol_target * mask[:,:,0]).sum() / mask.sum()

        entropies_ = entropies[:, :-1].reshape(-1, self.args.n_agents)
        target_entropy = th.ones_like(entropies_)
        for i in range(self.n_agents):
            target_entropy[:, i] = target_entropy[:, i] * self.target_entropy_sum[i]
        alpha_backup = target_entropy - entropies_
        alpha_loss = -(self.log_alpha * alpha_backup.detach()).sum(dim=1).mean()

        # Optimise
        self.p_optimiser.zero_grad()
        policy_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.alpha_params, self.args.grad_norm_clip)
        self.alpha_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm.cpu(), t_env)

            for i in range(self.n_agents):
                self.logger.log_stat("alpha" + str(i), alpha[i].cpu(), t_env)
                self.logger.log_stat("target_entropy_sum" + str(i), self.target_entropy_sum[i], t_env)

            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.alpha = th.exp(self.log_alpha)


        masked_td_error1_ent, masked_td_error2_ent = self.train_critic(batch, t_env)
        self.train_actor(batch, t_env, episode_num, masked_td_error1_ent=masked_td_error1_ent, masked_td_error2_ent=masked_td_error2_ent)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            if self.args.target_update_soft:
                self._update_targets_soft(0.001)
            else:
                self._update_targets()

            self.last_target_update_episode = episode_num

    def train_critic(self, batch, t_env):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        states = batch["state"]

        mac = self.mac
        mixer1 = self.mixer1
        mixer2 = self.mixer2
        alpha = self.alpha.detach()
        alpha = th.clamp(alpha, max=alpha_max, min=alpha_min)
        
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions
        mac_out[avail_actions == 0] = 0.0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        t_mac_out = mac_out.clone().detach() 
        pi = t_mac_out
        log_pi = th.log(pi)

        # sample actions for next timesteps
        next_actions = Categorical(pi).sample().long().unsqueeze(3)
        next_actions_onehot = th.zeros(next_actions.squeeze(3).shape + (self.n_actions,))
        if self.args.use_cuda:
            next_actions_onehot = next_actions_onehot.cuda()
        next_actions_onehot = next_actions_onehot.scatter_(3, next_actions, 1)

        pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)[:,1:]
        pi_taken[mask.expand_as(pi_taken) == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
        target_q_vals1_ = self.target_critic1.forward(target_inputs).detach()
        target_q_vals2_ = self.target_critic2.forward(target_inputs).detach()
        target_v_vals1 = th.sum(target_q_vals1_ * pi, dim=-1)
        target_v_vals2 = th.sum(target_q_vals2_ * pi, dim=-1)
        target_v_vals1 = self.target_mixer1(target_v_vals1, states)
        target_v_vals2 = self.target_mixer2(target_v_vals2, states)
        target_v_tot = th.min(target_v_vals1, target_v_vals2)

        target_q_vals1_ent_ = self.critic1_ent.forward(target_inputs)
        target_q_vals2_ent_ = self.critic2_ent.forward(target_inputs)
        target_v_vals1_ent = th.sum(target_q_vals1_ent_ * mac_out, dim=-1)
        target_v_vals2_ent = th.sum(target_q_vals2_ent_ * mac_out, dim=-1)
        target_v_vals1_ent = self.mixer1_ent(target_v_vals1_ent, states)
        target_v_vals2_ent = self.mixer2_ent(target_v_vals2_ent, states)
        target_v_ent_tot = th.min(target_v_vals1_ent, target_v_vals2_ent)

        entropies = - (pi * log_pi).sum(dim=-1)
        target_v_ent_tot = target_v_ent_tot + entropies * alpha

        if self.args.use_td_lambda:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_v_tot, self.n_agents, self.args.gamma,self.args.td_lambda)  ##  rewards
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_v_tot, self.n_agents, self.args.gamma,
                                              self.args.td_lambda)
        targets_ent = []
        for i in range(self.n_agents):
            if self.args.use_td_lambda:
                targets_ent_ = build_td_lambda_targets(th.zeros_like(rewards), terminated, mask, th.unsqueeze(target_v_ent_tot[:, :, i], -1), self.n_agents, self.args.gamma,
                                                  self.args.td_lambda)  ##  rewards
            else:
                targets_ent_ = self.args.gamma * th.unsqueeze(target_v_ent_tot[:, 1:, i], -1) * (1 - terminated)
            targets_ent.append(targets_ent_)


        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)
        q_taken1 = th.gather(q_vals1[:, :-1], dim=3, index=actions).squeeze(3)
        q_taken2 = th.gather(q_vals2[:, :-1], dim=3, index=actions).squeeze(3)
        q_taken1 = mixer1(q_taken1, states[:, :-1])
        q_taken2 = mixer2(q_taken2, states[:, :-1])

        td_error1 = q_taken1 - targets.detach()
        td_error2 = q_taken2 - targets.detach()

        q_vals1_ent = self.critic1_ent.forward(inputs)
        q_vals2_ent = self.critic2_ent.forward(inputs)

        q_taken1_ent = th.gather(q_vals1_ent[:, :-1], dim=3, index=actions).squeeze(3)
        q_taken2_ent = th.gather(q_vals2_ent[:, :-1], dim=3, index=actions).squeeze(3)

        q_taken1_ent = self.mixer1_ent(q_taken1_ent, states[:, :-1])
        q_taken2_ent = self.mixer2_ent(q_taken2_ent, states[:, :-1])


        mask = mask.expand_as(td_error1)

        loss1_ent, loss2_ent = 0, 0
        masked_td_error1_ent, masked_td_error2_ent = [], []
        for i in range(self.n_agents):
            td_error1_ent_ = (th.unsqueeze(q_taken1_ent[:, :, i], -1) - targets_ent[i].detach()) * mask
            td_error2_ent_ = (th.unsqueeze(q_taken2_ent[:, :, i], -1) - targets_ent[i].detach()) * mask
            loss1_ent = loss1_ent + (td_error1_ent_ ** 2).sum() / mask.sum()
            loss2_ent = loss2_ent + (td_error2_ent_ ** 2).sum() / mask.sum()
            masked_td_error1_ent.append(abs(td_error1_ent_.detach() ))
            masked_td_error2_ent.append(abs(td_error2_ent_.detach() ))

        # 0-out the targets that came from padded data
        masked_td_error1 = td_error1 * mask
        loss1 = (masked_td_error1 ** 2).sum() / mask.sum() 
        masked_td_error2 = td_error2 * mask
        loss2 = (masked_td_error2 ** 2).sum() / mask.sum() 
        
        # Optimise
        self.c_optimiser1.zero_grad()
        loss1.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
        self.c_optimiser1.step()
        
        self.c_optimiser2.zero_grad()
        loss2.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.grad_norm_clip)
        self.c_optimiser2.step()

        self.c_ent_optimiser1.zero_grad()
        loss1_ent.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_ent_params1, self.args.grad_norm_clip)
        self.c_ent_optimiser1.step()

        self.c_ent_optimiser2.zero_grad()
        loss2_ent.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_ent_params2, self.args.grad_norm_clip)
        self.c_ent_optimiser2.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss1.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)

        if self.args.tderror_exploration:
            return masked_td_error1_ent, masked_td_error2_ent
        else:
            return None, None

    def _update_targets(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_mixer1.load_state_dict(self.mixer1.state_dict())
        self.target_mixer2.load_state_dict(self.mixer2.state_dict())

        self.target_critic1_ent.load_state_dict(self.critic1_ent.state_dict())
        self.target_critic2_ent.load_state_dict(self.critic2_ent.state_dict())
        self.target_mixer1_ent.load_state_dict(self.mixer1_ent.state_dict())
        self.target_mixer2_ent.load_state_dict(self.mixer2_ent.state_dict())


        self.logger.console_logger.info("Updated target network")

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mixer1_ent.parameters(), self.mixer1_ent.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_mixer2_ent.parameters(), self.mixer2_ent.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic1_ent.parameters(), self.critic1_ent.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic2_ent.parameters(), self.critic2_ent.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_mixer1.parameters(), self.mixer1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_mixer2.parameters(), self.mixer2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic1.cuda()
        self.mixer1.cuda()
        self.target_critic1.cuda()
        self.target_mixer1.cuda()
        self.critic2.cuda()
        self.mixer2.cuda()
        self.target_critic2.cuda()
        self.target_mixer2.cuda()

        self.critic1_ent.cuda()
        self.mixer1_ent.cuda()
        self.target_critic1_ent.cuda()
        self.target_mixer1_ent.cuda()
        self.critic2_ent.cuda()
        self.mixer2_ent.cuda()
        self.target_critic2_ent.cuda()
        self.target_mixer2_ent.cuda()



    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.mixer1.state_dict(), "{}/mixer1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        th.save(self.mixer2.state_dict(), "{}/mixer2.th".format(path))
        th.save(self.p_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.c_optimiser1.state_dict(), "{}/critic_opt1.th".format(path))
        th.save(self.c_optimiser2.state_dict(), "{}/critic_opt2.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer2.load_state_dict(th.load("{}/mixer2.th".format(path), map_location=lambda storage, loc: storage))

        self.p_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser1.load_state_dict(th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser2.load_state_dict(th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))
        
    def build_inputs(self, batch, bs, max_t, actions_onehot):
        inputs = []
        inputs.append(batch["obs"][:])
        actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
        inputs.append(actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
