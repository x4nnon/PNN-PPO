#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:28:51 2024

@author: x4nno
"""

import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import time
from torch.cuda.amp import autocast

from transformers import SwinModel


preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert to a PyTorch tensor and scale to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet's mean and std
])

from transformers import ViTFeatureExtractor, ViTModel
import torchvision.transforms as transforms

model_name = 'google/vit-base-patch16-224'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(tensor):
    norm = torch.linalg.norm(tensor)
    return tensor / norm if norm != 0 else tensor


# Function to preprocess a single image
def preprocess_image(image_array, mean=None, std=None):
    # Convert the NumPy array to a PIL image
    image = image_array.permute(0, 3, 1, 2)
    image = F.interpolate(image, 224)
    image = image/255 # this scales
    
    # these are from the ResNet
    if mean == None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    image = (image - mean) / std
    
    # image = Image.fromarray(image_array.astype('uint8'))
    # Apply the preprocessing pipeline
    return image

    
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ProcGenActor(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_dim):
        super(ProcGenActor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, state):
        x = F.relu(self.conv1(state), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = F.relu(self.conv3(x), inplace=False)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x), inplace=False)
        x = self.fc2(x)
        return x
    
    def get_intermediate_features(self, state):
        """Returns the intermediate convolutional features before flattening."""
        x = F.relu(self.conv1(state), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = F.relu(self.conv3(x), inplace=False)
        return x 
    
    def forward_from_features(self, features):
        """Pass the intermediate features through the fully connected layers to get final output."""
        # x = features.view(features.size(0), -1)  # Flatten the features
        x = features.reshape(features.size(0), -1)
        x = F.relu(self.fc1(x), inplace=False)
        return self.fc2(x)  # Final logits
    
    
class ProcGenCritic(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(ProcGenCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state): #, action):
        x = F.relu(self.conv1(state), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = F.relu(self.conv3(x), inplace=False)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        
        # x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc1(x), inplace=False)
        q_value = self.fc2(x)
        return q_value
    
    def get_intermediate_features(self, state):
        """Returns the intermediate convolutional features before flattening."""
        x = F.relu(self.conv1(state), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = F.relu(self.conv3(x), inplace=False)
        return x 
    
    def forward_from_features(self, features):
        """
        Pass the intermediate features through the fully connected layers to get the final value output.
        :param features: 4D feature map from the convolutional layers.
        :return: A scalar value estimate for the current state.
        """
        x = features.reshape(features.size(0), -1)
        # x = features.view(features.size(0), -1)  # Flatten the features
        x = F.relu(self.fc1(x), inplace=False)
        return self.fc2(x)  # Final scalar value


def show_sim_states(good_obs_tensor_filt, next_obs, action, ms):
    
    plt.imshow(next_obs/255)
    plt.title("current obs")
    plt.show()
    plt.imshow(np.array(good_obs_tensor_filt.cpu())/255)
    plt.title(f"{action}, {ms}")
    plt.show()
    


class PNNAdapterLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, padding=0):
        """
        Adapter layer to transform inputs from previous columns for use in current column.
        Uses convolutional layers to preserve spatial structure.
        """
        super(PNNAdapterLayer, self).__init__()
        self.adapter_layer = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return F.relu(self.adapter_layer(x), inplace=False)



class PNNAgent(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_dim, num_columns):
        """
        Initialize the PNN Agent with convolutional lateral connections (adapters).
        :param input_channels: Number of input channels (e.g., for images).
        :param num_actions: Number of possible actions in the environment.
        :param hidden_dim: The size of the hidden layer in the actor-critic network.
        :param num_columns: Number of columns (tasks) in the PNN.
        """
        super(PNNAgent, self).__init__()

        self.num_columns = num_columns  # Number of tasks/columns

        # Store the actor and critic for each task
        self.actors = nn.ModuleList([ProcGenActor(input_channels, num_actions, hidden_dim).to(device) for _ in range(num_columns)])
        self.critics = nn.ModuleList([ProcGenCritic(input_channels, hidden_dim).to(device) for _ in range(num_columns)])

        # Store convolutional adapter layers to enable knowledge transfer between columns
        self.actor_adapters = nn.ModuleList()
        self.critic_adapters = nn.ModuleList()

        for col in range(1, num_columns):  # Adapters between columns
            actor_adapter = nn.ModuleList([
                PNNAdapterLayer(64, 64) for _ in range(col)  # Assuming the output from each actor is 64 channels
            ])
            critic_adapter = nn.ModuleList([
                PNNAdapterLayer(64, 64) for _ in range(col)
            ])
            self.actor_adapters.append(actor_adapter.to(device))
            self.critic_adapters.append(critic_adapter.to(device))

    def get_previous_outputs(self, state, current_column):
        """Get previous outputs from actor and critic, including intermediate features."""
        if current_column == 0:
            return None, None

        previous_actor_outputs = []
        previous_critic_outputs = []

        for col in range(current_column):
            actor_output, actor_features = self.forward_actor(state, col, previous_actor_outputs)
            previous_actor_outputs.append((actor_features, actor_output))

            critic_output, critic_features = self.forward_critic(state, col, previous_critic_outputs)
            previous_critic_outputs.append((critic_features, critic_output))

        return previous_actor_outputs, previous_critic_outputs
    
    
    def forward_actor(self, state, column, previous_outputs=None):
        """Forward pass for actor, returns both intermediate features and final output."""
        # Pass through the actor network
        intermediate_features = self.actors[column].get_intermediate_features(state)  # Assuming you modify the actor to return intermediate features
          # Final logits from actor

        # Apply adapters to intermediate features from previous columns (if any)
        if column > 0 and previous_outputs:
            for i in range(column):
                # Apply convolutional adapters to intermediate features
                prev_features = previous_outputs[i][0]  # Previous intermediate features
                adapter_output = self.actor_adapters[column - 1][i](prev_features)
                intermediate_features = torch.add(intermediate_features, adapter_output)
                # intermediate_features += adapter_output ## <---- THIS IS INPLACE DON@T DO THIS in PYTROCH!

        actor_output = self.actors[column].forward_from_features(intermediate_features)
        # Return the final actor output (logits) and intermediate features
        return actor_output, intermediate_features

    def forward_critic(self, state, column, previous_outputs=None):
        """Forward pass for critic, returns both intermediate features and final value."""
        # Pass through the critic network
        intermediate_features = self.critics[column].get_intermediate_features(state)  # Assuming you modify the critic to return intermediate features
          # Final value from critic

        # Apply adapters to intermediate features from previous columns (if any)
        if column > 0 and previous_outputs:
            for i in range(column):
                # Apply convolutional adapters to intermediate features
                prev_features = previous_outputs[i][0]  # Previous intermediate features
                adapter_output = self.critic_adapters[column - 1][i](prev_features)
                intermediate_features = torch.add(intermediate_features, adapter_output)
                # intermediate_features += adapter_output ##<----- in place operations BAD BAD BAD 
        critic_output = self.critics[column].forward_from_features(intermediate_features)
        # Return the final critic output (value) and intermediate features
        return critic_output, intermediate_features
    

    def get_value(self, state, column, previous_critic_outputs=None):
        """
        Get the value (critic output) for a given state and column (task).
        :param state: Input state.
        :param column: The current column/task.
        :param previous_critic_outputs: Outputs from previous columns' critics.
        :return: Value estimate.
        """
        state = state.permute(0,3,1,2)
        state = state/255
        
        value, _ = self.forward_critic(state, column, previous_critic_outputs)
        
        return value

    def get_action_and_value(self, state, current_column, action=None, previous_actor_outputs=None, previous_critic_outputs=None):
        """
        Get the action and value for a given state and column (task).
        :param state: Input state.
        :param column: The current column/task.
        :param action: (Optional) Action to evaluate.
        :param previous_actor_outputs: Outputs from previous columns' actors.
        :param previous_critic_outputs: Outputs from previous columns' critics.
        :return: Action, log_prob, entropy, value.
        """
        # Actor forward pass to get action logits
        
        state = state.permute(0,3,1,2)
        state = state/255
        previous_actor_outputs, previous_critic_outputs = self.get_previous_outputs(state, current_column)
        
        logits, _ = self.forward_actor(state, current_column, previous_actor_outputs)

        # Create a categorical distribution from the logits for discrete actions
        dist = Categorical(logits=logits)
        
        # Sample an action if none is provided
        if action is None:
            action = dist.sample()

        # Get the critic value
        value, _ = self.forward_critic(state, current_column, previous_critic_outputs)

        return action, dist.log_prob(action), dist.entropy(), value


