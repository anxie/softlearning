from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm

from softlearning.models.feedforward import feedforward_model

def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            target_kl=0.5,
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter

        self._encoder_lr = 0.005
        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._latent_dim = 2
        self._max_episodes = 4000
        self._prev_q_map = np.zeros((self._max_episodes, self._latent_dim))
        # self._prev_q_map[0] = np.array([0.1 * np.cos(-0.5), 0.1 * np.sin(-0.5)])
        for t in range(20):
            self._prev_q_map[t] = 0.1*np.array([np.cos(0.5*(t-1)), np.sin(0.5*(t-1))])
        self._q_map_update_interval = 1000
        self._target_kl = 1e-4 #target_kl

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize

        self._save_full_state = save_full_state

        self._build()

    def _build(self):
        super(SAC, self)._build()

        self._init_encoder_update()
        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _get_Q_target(self):
        observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        }
        policy_inputs = flatten_input_structure(
            {**observations, 'env_latents': self.next_latents})

        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_observations = flatten_input_structure(
            {**next_Q_observations, 'env_latents': self.next_latents})
        next_Q_inputs = flatten_input_structure(
            {'observations': next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._placeholders['rewards'],
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = self._get_Q_target()
        assert Q_target.shape.as_list() == [None, 1]

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_observations = flatten_input_structure(
            {**Q_observations, 'env_latents': self.latents})
        Q_inputs = flatten_input_structure({
            'observations': Q_observations, 'actions': self._placeholders['actions']})
        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        observations = {
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        }
        policy_inputs = flatten_input_structure(
            {**observations, 'env_latents': self.latents})
        actions = self._policy.actions(policy_inputs)
        log_pis = self._policy.log_pis(policy_inputs, actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_observations = flatten_input_structure(
            {**Q_observations, 'env_latents': self.latents})
        Q_inputs = flatten_input_structure({
            'observations': Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_encoder_update(self):
        observations = {
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        }
        next_observations = {
            'next_{}'.format(name): self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        }
        encoder_inputs = flatten_input_structure({
            **observations,
            **next_observations,
            'actions': self._placeholders['actions'],
            'rewards': self._placeholders['rewards']})

        encoder_inputs = tf.concat(encoder_inputs, axis=-1)

        self.encoder_net = feedforward_model(
            hidden_layer_sizes=(64, 64),
            output_size=2 * self._latent_dim)
        params = self.encoder_net(encoder_inputs)
        mu = params[:, :self._latent_dim]
        log_sigma_sq = params[:, self._latent_dim:]
        eps = tf.random.normal(shape=tf.shape(mu))
        self.latents = self.next_latents = mu #+ eps * tf.sqrt(tf.exp(log_sigma_sq))

        with tf.variable_scope('latent'):
            init = np.random.uniform(0.82, 0.92)
            print('RANDOM INITIALIZATON:', init)
            cos_prior = tf.compat.v1.get_variable('cos_prior',
                initializer=tf.cast(np.arctanh(init), dtype=tf.float32))
            tanh_cos_prior = tf.tanh(cos_prior)
            sin_prior = tf.sqrt(1.0 - tf.square(tanh_cos_prior))
            self._dynamics_prior = tf.stack([[tanh_cos_prior, -sin_prior], [sin_prior, tanh_cos_prior]],
                name='dynamics_prior')

        #     ts = tf.reshape(tf.cast(self._placeholders['meta_times'], tf.int64), [-1])
        #     z0 = tf.cast(np.array([0.1, 0.0]), tf.float32)
        #     self._latent_priors = [z0]
        #     curr_latent = z0
        #     for i in range(1, self._max_episodes):
        #         curr_latent = tf.linalg.matvec(self._dynamics_prior, curr_latent)
        #         self._latent_priors.append(curr_latent)
        #     self._latent_priors = tf.stack(self._latent_priors, name='latent_priors')
        
        # latent_prior_a = tf.gather(self._latent_priors, ts)

        prev_latents = self._placeholders['prev_latents']
        tiled_dynamics_prior = tf.tile(tf.expand_dims(self._dynamics_prior, 0), [tf.shape(prev_latents)[0], 1, 1])
        latent_prior = tf.linalg.matvec(tiled_dynamics_prior, prev_latents)

        # encoder_kl_losses = -0.5 * (1 + log_sigma_sq - tf.square(mu - latent_prior) - tf.exp(log_sigma_sq))
        encoder_kl_losses = tf.square(mu - latent_prior)
        self._encoder_losses = encoder_kl_losses
        encoder_loss = tf.reduce_mean(encoder_kl_losses)

        beta = self._beta = tf.compat.v1.get_variable(
            'beta',
            dtype=tf.float32,
            initializer=1.0)

        if isinstance(self._target_kl, Number):
            beta_loss = tf.reduce_mean(
                beta * tf.stop_gradient(self._target_kl - encoder_loss))

            self._beta_optimizer = tf.compat.v1.train.AdamOptimizer(
                1e-3, name='beta_optimizer')
            self._beta_train_op = self._beta_optimizer.minimize(
                loss=beta_loss, var_list=[beta])

            self._training_ops.update({
                'temperature_beta': self._beta_train_op
            })

        self._encoder_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._encoder_lr,
            name="encoder_optimizer")

        encoder_train_op = self._encoder_optimizer.minimize(
            beta*encoder_loss,
            var_list=self.encoder_net.trainable_variables)

        # initial_learning_rate = 0.5
        # lr_schedule = tf.compat.v1.train.exponential_decay(
        #     initial_learning_rate,
        #     self._placeholders['iteration'],
        #     decay_steps=1000,
        #     decay_rate=0.5,
        #     staircase=True)

        # self._prior_optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        #     learning_rate=lr_schedule,
        #     name="prior_optimizer")

        self._prior_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.01,
            name="prior_optimizer")

        prior_train_op = self._prior_optimizer.minimize(
            beta*encoder_loss+1e-8*tf.square(cos_prior),
            var_list=[cos_prior])

        self._training_ops.update({'encoder_train_op': encoder_train_op})
        self._training_ops.update({'prior_train_op': prior_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('alpha', self._alpha),
            ('encoder_loss', self._encoder_losses),
            ('beta', self._beta),
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _update_q_map(self, encs, meta_times):
        meta_times = np.squeeze(meta_times)
        min_t, max_t = np.min(meta_times), np.max(meta_times)
        for t in range(min_t, max_t+1):
            if t < 20:
                continue
            enc_t = encs[np.where(meta_times == t)]
            if enc_t.shape[0] == 0:
                continue
            enc_t = np.mean(enc_t, 0)
            self._prev_q_map[t+1] = enc_t

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)

        if iteration % self._q_map_update_interval == 0:
            enc = self._session.run(self.latents, feed_dict)
            self._update_q_map(enc, batch['meta_times'])

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        ts = np.squeeze(batch_flat[('meta_times',)])
        feed_dict[self._placeholders['prev_latents']] = self._prev_q_map[ts]

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)

        observations = {
            name: batch['observations'][name]
            for name in self._policy.observation_keys
        }
        ts = np.squeeze(batch['meta_times']).astype(np.int64)
        latents = self._session.run(self.latents, feed_dict)
        inputs = flatten_input_structure({
            **observations,
            'env_latents': latents
        })

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(inputs).items()
        ]))

        A = self._session.run(self._dynamics_prior)
        diagnostics.update(OrderedDict([
            ('cos_prior', A[0,0])
        ]))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
