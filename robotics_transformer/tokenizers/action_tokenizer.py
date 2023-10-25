# -*- coding: utf-8 -*-
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple action tokenizer used with Robotics Transformer 1.

As an example, if an action is: terminate = [0, 1] world_vector = [0.9,
0.8, -0.3] rotation_delta = [-0.1, 0.2, .6] gripper_closedness = 0.9

Then we build a sequence of tokens of length 8 [one for each dimension].
The int32 type action dimensions are already assumed discrete and
tokenized, the float dimensions are bucketed according to the specs min
and max. Each dimension has 'vocab_size' buckets.

Currently, this tokenizer assumes one action spec and it is highly
recommended to specify the 'action_order', eg [terminate, world_vector,
rotation_delta, gripper_closedness]. Since after tokenization you lose
that information, this will be useful for debugging. Actions may also be
subselected for prediction, since not all actions are needed in the
action_order.
"""
from typing import Optional

import tensorflow as tf

from tensor2robot.utils import tensorspec_utils


class RT1ActionTokenizer:
    """Tokenizes based on vocab size."""

    def __init__(self,
                 action_spec: tensorspec_utils.TensorSpecStruct,
                 vocab_size: int,
                 action_order: Optional[list[str]] = None):
        """Instantiates an RT1ActionTokenizer.

        Args:
          action_spec: Tensor spec of the expected action tensor.
          vocab_size: Number of buckets to discretize action to.
          action_order: Order of the action names, used to discern the order of
            tokenized actions to detokenize and assemble back to action tensor
        """
        self._action_spec = action_spec
        self._vocab_size = vocab_size
        if action_order is None:
            self._action_order = self._action_spec.keys()
        else:
            for action in action_order:
                if action not in self._action_spec.keys():
                    raise ValueError(
                        f'actions: {action} not found in action_spec: {action_spec.keys()}'
                    )
                assert action in self._action_spec.keys()
            self._action_order = action_order
        self._tokens_per_action = 0
        for action in self._action_order:
            action_shape = self._action_spec[action].shape
            if len(action_shape) != 1:
                raise ValueError(
                    f'Only action shapes with single dimension supported, got {action_shape}'
                )
            if self._action_spec[action].dtype == tf.int32:
                # Int32 actions are already assumed to be tokens.
                self._tokens_per_action += 1
            else:
                self._tokens_per_action += action_shape[0]

        num_action_token = sum(
            1 if spec.dtype == tf.int32 else spec.shape[-1]
            for spec in self._action_spec.values()
        )
        tf.debugging.assert_equal(num_action_token, self._tokens_per_action)
        tf.debugging.assert_equal(num_action_token, self._tokens_per_action)

    @property
    def tokens_per_action(self) -> int:
        return self._tokens_per_action

    @property
    def action_spec(self) -> tensorspec_utils.TensorSpecStruct:
        return self._action_spec

    @property
    def action_order(self) -> list[str]:
        return self._action_order

    def tokenize(self, action: tensorspec_utils.TensorSpecStruct) -> tf.Tensor:
        """Tokenizes an action."""
        action_tokens = []
        for k in self._action_order:
            a = action[k]  # a is [batch, actions_size]
            spec = self._action_spec[k]
            if spec.dtype == tf.int32:
                # Int32 actions are already assumed to be tokens, assume it is smaller
                # than the vocab size, so all we need to do is pad zeros.
                tf.debugging.assert_equal(1, tf.reduce_sum(a, axis=-1))
                # extract the token [batch, 1]
                token = tf.argmax(a, axis=-1, output_type=tf.int32)
                tf.debugging.assert_less(token, self._vocab_size)
                # Add a seq dimension [batch, 1]
                token = tf.expand_dims(token, axis=-1)
            else:
                a = tf.clip_by_value(a, spec.minimum, spec.maximum)
                # Normalize the action [batch, actions_size]
                token = (a - spec.minimum) / (spec.maximum - spec.minimum)
                # Bucket and discretize the action to vocab_size, [batch, actions_size]
                token = tf.cast(token * (self._vocab_size - 1), tf.int32)
            action_tokens.append(token)
        return tf.concat(action_tokens, axis=-1)

    def detokenize(self,
                   action_tokens: tf.Tensor) -> tensorspec_utils.TensorSpecStruct:
        """Detokenizes an action."""
        action = tensorspec_utils.TensorSpecStruct()
        token_index = 0
        for k in self._action_order:
            spec = self._action_spec[k]
            action_dim = spec.shape[0]
            if spec.dtype == tf.int32:
                # Int32 actions are already assumed to be tokens.
                action[k] = action_tokens[..., token_index]
                # A poor model may output tokens outside the allowed range, in that case
                # set them to a default value, the 0 token in this case.
                outside_range = tf.greater_equal(action[k], action_dim)
                action[k] = tf.where(
                    outside_range, tf.zeros_like(action[k]), action[k])
                action[k] = tf.one_hot(
                    action[k], depth=action_dim, axis=-1, dtype=tf.int32)
                token_index += 1
            else:
                actions = []
                for _ in range(action_dim):
                    a = action_tokens[..., token_index:token_index + 1]
                    a = tf.cast(a, tf.float32)
                    a = a / (self._vocab_size - 1)
                    a = (a * (spec.maximum - spec.minimum)) + spec.minimum
                    actions.append(a)
                    token_index += 1
                action[k] = tf.concat(actions, axis=-1)
        return action
