class NADE(nn.Module):
    """
    An implementation of a Neural Autoregressive Distribution Estimator (NADE).
    """
    def __init__(self, input_size, internal_size=512):
        super().__init__()
        self.input_size = input_size
        self.internal_size = internal_size

        # Assign degrees to neurons
        # Input degrees are their indices: 0, 1, ..., D-1
        self.m_input = torch.arange(self.input_size)
        
        # Hidden layer degrees are sampled from 0 to D-2
        # This ensures any hidden unit can't see the last input x_{D-1}
        self.m_hidden = torch.from_numpy(
            np.random.randint(0, self.input_size - 1, size=self.internal_size)
        ).sort()[0]

        # Define layers
        self.fc1 = nn.Linear(input_size, internal_size)
        self.fc2 = nn.Linear(internal_size, input_size)
        
        # Create masks
        # Mask 1: input -> hidden
        # Connection is allowed if hidden unit degree >= input unit degree
        mask1 = (self.m_hidden.unsqueeze(1) >= self.m_input.unsqueeze(0)).float()
        
        # Mask 2: hidden -> output
        # Connection is allowed if output unit degree >= hidden unit degree
        # Note: output degrees are the same as input degrees
        mask2 = (self.m_input.unsqueeze(1) >= self.m_hidden.unsqueeze(0)).float()

        # Register masks as non-trainable buffers
        self.register_buffer("mask1", mask1)
        self.register_buffer("mask2", mask2)

        self.activation = nn.Tanh()

    def forward(self, x):
        """
        A single, efficient forward pass.
        """
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input of size {self.input_size}, got {x.shape[1]}")

        # Apply masks to the weights during the forward pass
        h = self.activation(F.linear(x, self.fc1.weight * self.mask1, self.fc1.bias))
        logits = F.linear(h, self.fc2.weight * self.mask2, self.fc2.bias)
        
        return logits

    def sample(self, num_samples):
        """
        Autoregressive sampling. This part is inherently sequential.
        """
        device = self.fc1.weight.device
        samples = torch.zeros(num_samples, self.input_size, device=device)
        
        for i in range(self.input_size):
            logits = self.forward(samples)
            prob = torch.sigmoid(logits[:, i])
            samples[:, i] = torch.bernoulli(prob)
            
        return samples

    def evalP(self, x):
        """
        Vectorized probability evaluation of a given batch of data x.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        # Single forward pass to get all conditional logits
        logits = self.forward(x)
        prob_vec = torch.sigmoid(logits)

        # Calculate the log probability using the formula for Bernoulli distributions
        log_p_x = torch.sum(
            x * F.logsigmoid(logits) + (1 - x) * F.logsigmoid(-logits),
            dim=1
        )
        p_x = torch.exp(log_p_x)
        
        return p_x.squeeze(), log_p_x.squeeze()

    def fill(self, x_partial, mask=None):
        """
        Fill missing pixels in an image using the NADE model.
        Args:
            x_partial: [input_size] or [1, input_size], known pixels set, unknowns = 0 or any value.
            mask: same shape, 1=known, 0=unknown. If None, treat zeros as unknowns.
        Returns:
            x_filled: completed image
        """
        if x_partial.ndim == 1:
            x_partial = x_partial.unsqueeze(0)

        batch_size, input_size = x_partial.shape
        x_filled = x_partial.clone()

        if mask is None:
            mask = (x_partial != 0).float()

        for i in range(input_size):
            for b in range(batch_size):
                if mask[b, i] == 0:  # unknown pixel
                    logits = self.forward(x_filled[b:b+1])
                    prob = torch.sigmoid(logits)[0, i]
                    x_filled[b, i] = torch.bernoulli(prob)

        return x_filled.squeeze(0) if batch_size == 1 else x_filled
