import numpy as np

class ReplayBuffer():
    def __init__(self, size, input_shape, n_actions) -> None:
        self.size = size
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.mem_counter = 0 # keeps track of the first available memory index
        self.state_mem = np.zeros((self.size, *self.input_shape))
        self.n_state_mem = np.zeros((self.size, *self.input_shape))
        self.action_mem = np.zeros((self.size, self.n_actions))

        self.reward_mem = np.zeros((self.size)) # scalar values
        self.terminal_mem = np.zeros(self.size, dtype=np.bool)

    def store_transition(self, state, action, reward, n_state, done):
        idx = self.mem_counter % self.size

        # Add the transition experience to memory
        self.state_mem[idx] = state
        self.n_state_mem[idx] = n_state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = done

        # Increment the counter
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.size)

        # get a randomized batch of samples
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_mem[batch]
        n_states = self.n_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]

        return states, actions, rewards, n_states, dones