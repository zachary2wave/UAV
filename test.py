for episode in range(nb_episodes):
    callbacks.on_episode_begin(episode)
    episode_reward = 0.
    episode_step = 0

    # Obtain the initial observation by resetting the environment.
    self.reset_states()
    observation = deepcopy(env.reset())
    if self.processor is not None:
        observation = self.processor.process_observation(observation)
    assert observation is not None

    # Perform random starts at beginning of episode and do not record them into the experience.
    # This slightly changes the start position between games.
    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
    for _ in range(nb_random_start_steps):
        if start_step_policy is None:
            action = env.action_space.sample()
        else:
            action = start_step_policy(observation)
        if self.processor is not None:
            action = self.processor.process_action(action)
        callbacks.on_action_begin(action)
        observation, r, done, info = env.step(action)
        observation = deepcopy(observation)
        if self.processor is not None:
            observation, r, done, info = self.processor.process_step(observation, r, done, info)
        callbacks.on_action_end(action)
        if done:
            warnings.warn(
                'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                    nb_random_start_steps))
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            break

    # Run the episode until we're done.
    done = False
    while not done:
        callbacks.on_step_begin(episode_step)

        action = self.forward(observation)
        if self.processor is not None:
            action = self.processor.process_action(action)
        reward = 0.
        accumulated_info = {}
        for _ in range(action_repetition):
            callbacks.on_action_begin(action)
            observation, r, d, info = env.step(action)
            observation = deepcopy(observation)
            if self.processor is not None:
                observation, r, d, info = self.processor.process_step(observation, r, d, info)
            callbacks.on_action_end(action)
            reward += r
            for key, value in info.items():
                if not np.isreal(value):
                    continue
                if key not in accumulated_info:
                    accumulated_info[key] = np.zeros_like(value)
                accumulated_info[key] += value
            if d:
                done = True
                break
        if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
            done = True
        self.backward(reward, terminal=done)
        episode_reward += reward

        step_logs = {
            'action': action,
            'observation': observation,
            'reward': reward,
            'episode': episode,
            'info': accumulated_info,
        }
        callbacks.on_step_end(episode_step, step_logs)
        episode_step += 1
        self.step += 1

    # We are in a terminal state but the agent hasn't yet seen it. We therefore
    # perform one more forward-backward call and simply ignore the action before
    # resetting the environment. We need to pass in `terminal=False` here since
    # the *next* state, that is the state of the newly reset environment, is
    # always non-terminal by convention.
    self.forward(observation)
    self.backward(0., terminal=False)
    ################################
    # env.trajectory()
    # basic = env.putout()  # drz
    ################################
    # Report end of episode.
    episode_logs = {
        'episode_reward': episode_reward,
        'nb_steps': episode_step,
        # 'episode_basic_data': basic
    }
    callbacks.on_episode_end(episode, episode_logs)