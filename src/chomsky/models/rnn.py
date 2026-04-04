# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builders for RNN/LSTM cores."""
import abc
from typing import Any, Tuple
import chex

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax
# from chomsky.models import tape_rnn

class RNNCell(eqx.Module, abc.ABC):

  @abc.abstractmethod
  def step(self, x: chex.Array, state: Any, *, key=None) -> Tuple[chex.Array, Any]: ...

  @property
  @abc.abstractmethod
  def output_size(self) -> int: ...

  @abc.abstractmethod
  def initial_state(self) -> Any: ...

class LSTM(eqx.nn.LSTMCell, RNNCell):
   
  @property
  def output_size(self) -> int: return self.hidden_size

  def initial_state(self) -> Any:
    return (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))
  
  def step(self, x: chex.Array, state: Any, *, key=None) -> Tuple[chex.Array, Any]:
      (h, c) = self(x, state, key=key)
      return h, (h, c)

class VanillaRNN(RNNCell):
    hidden_size: int
    input_to_hidden: eqx.nn.Linear
    hidden_to_hidden: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, *, key):
        keys = jrandom.split(key)
        self.hidden_size = hidden_size
        self.input_to_hidden = eqx.nn.Linear(input_size, hidden_size, key=keys[0])
        self.hidden_to_hidden = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[1])

    @property
    def output_size(self) -> int: return self.hidden_size

    def initial_state(self) -> Any:
       return jnp.zeros((self.hidden_size,))

    def step(self, x: chex.Array, state: Any, *, key=None) -> Tuple[chex.Array, Any]:
      carry = self(x, state, key=key)
      return carry, carry

    def __call__(self, x: chex.Array, state: Any, *, key=None) -> Any:
        out = jax.nn.relu(self.input_to_hidden(x) + self.hidden_to_hidden(state))
        return out, out

class RNN(eqx.Module):
  input_size: int = eqx.field(static=True)
  output_size: int = eqx.field(static=True)
  input_window: int = eqx.field(static=True)
  return_all_outputs: bool = eqx.field(static=True)

  core: RNNCell
  output_lin: eqx.nn.Linear

  def __init__(self, 
               key: chex.PRNGKey,
               input_size: int,
               output_size: int, 
               rnn_core: type[RNNCell],
               return_all_outputs: bool = False,
               input_window: int = 1,
               **rnn_kwargs):
    
    in_size = input_window * input_size
    self.input_window = input_window
    self.input_size = input_size
    self.output_size = output_size
    self.return_all_outputs = return_all_outputs

    keys = jrandom.split(key)
    self.core = rnn_core(input_size=in_size, **rnn_kwargs, key=keys[0])
    self.output_lin = eqx.nn.Linear(self.core.output_size, output_size, key=keys[1])

  def __call__(self, x: chex.Array, input_length: int = 1):
    # if issubclass(rnn_core, tape_rnn.TapeRNNCore):
    #   initial_state = self.core.initial_state(input_length)  # pytype: disable=wrong-arg-count
    # else:
    #   initial_state = self.core.initial_state()
    initial_state = self.core.initial_state()

    seq_length, _ = x.shape
    if seq_length % self.input_window != 0:
      x = jnp.pad(x, ((0, self.input_window - seq_length % self.input_window),
                      (0, 0)))
    new_seq_length = x.shape[0]
    x = jnp.reshape(
        x,
        (new_seq_length // self.input_window, -1))
        
    # x = hk.Flatten(preserve_dims=1)(x)
    # x = jnp.reshape(x, (new_seq_length // self.input_window, -1))

    def scan_fn(state, x):
      output, state = self.core.step(x, state)
      return state, output
    
    _, output = jax.lax.scan(scan_fn, initial_state, x)
    # output = jnp.reshape(output, (new_seq_length, output.shape[-1]))

    output = jnn.relu(output)
    if not self.return_all_outputs:
      output = output[-1, :]  # (time, alphabet_dim)
      return self.output_lin(output)
    else:
      return eqx.filter_vmap(self.output_lin)(output)
    
      
def make_rnn(
    key: chex.PRNGKey,
    input_size: int,
    output_size: int,
    rnn_core: type[eqx.Module],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **rnn_kwargs: Any
) -> eqx.Module:
  """Returns an RNN model.

  Only the last output in the sequence is returned. A linear layer is added to
  match the required output_size.

  Args:
    output_size: The output size of the model.
    rnn_core: The haiku RNN core to use. LSTM by default.
    return_all_outputs: Whether to return the whole sequence of outputs of the
      RNN, or just the last one.
    input_window: The number of tokens that are fed at once to the RNN.
    **rnn_kwargs: Kwargs to be passed to the RNN core.
  """

  return RNN(key=key, input_size=input_size, output_size=output_size, rnn_core=rnn_core, return_all_outputs=return_all_outputs, input_window=input_window, **rnn_kwargs)
