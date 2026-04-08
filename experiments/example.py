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

"""Example script to train and evaluate a network."""

from absl import app
from absl import flags
import jax.numpy as jnp
import numpy as np
import jax.random as jrandom

import constants
from chomsky import curriculum as curriculum_lib
import training
import utils

_SEED = flags.DEFINE_integer(
    'seed',
    default=0,
    help='Experiment seed.',
    lower_bound=0,
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    default=128,
    help='Training batch size.',
    lower_bound=1,
)
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length',
    default=40,
    help='Maximum training sequence length.',
    lower_bound=1,
)
_TASK = flags.DEFINE_string(
    'task',
    default='even_pairs',
    help='Length generalization task (see `constants.py` for other tasks).',
)
_ARCHITECTURE = flags.DEFINE_string(
    'architecture',
    default='rnn',
    help='Model architecture (see `constants.py` for other architectures).',
)

_IS_AUTOREGRESSIVE = flags.DEFINE_boolean(
    'is_autoregressive',
    default=False,
    help='Whether to use autoregressive sampling or not.',
)
_COMPUTATION_STEPS_MULT = flags.DEFINE_integer(
    'computation_steps_mult',
    default=0,
    help=(
        'The amount of computation tokens to append to the input tape (defined'
        ' as a multiple of the input length)'
    ),
    lower_bound=0,
)
# The architecture parameters depend on the architecture, so we cannot define
# them as via flags. See `constants.py` for the required values.
_ARCHITECTURE_PARAMS = {
    'hidden_size': 256,
}


def main(unused_argv) -> None:
  key = jrandom.key(_SEED.value)
  keys = jrandom.split(key)

  # Create the task.
  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(1, _SEQUENCE_LENGTH.value + 1))
  )
  task = constants.TASK_BUILDERS[_TASK.value]()

  # Create the model.
  single_output = task.output_length(10) == 1

  is_autoregressive = _IS_AUTOREGRESSIVE.value
  is_transformer = 'transformer' in _ARCHITECTURE.value
  extra_dims_onehot = 1 + int(_COMPUTATION_STEPS_MULT.value > 0)

  # Resolve input size based on architecture/mode combination
  if is_autoregressive and not is_transformer:
    input_size = max(task.input_size, task.output_size) + extra_dims_onehot
  elif is_autoregressive:  # transformer
    input_size = task.input_size
  else:
    input_size = task.input_size + extra_dims_onehot

  model = constants.MODEL_BUILDERS[_ARCHITECTURE.value](
    key=keys[0],
    input_size=input_size,
    output_size=task.output_size,
    return_all_outputs=True,
    **_ARCHITECTURE_PARAMS,
  )
  
  if is_autoregressive:
    if not is_transformer:
        model = utils.make_model_with_targets_as_input(model, _COMPUTATION_STEPS_MULT.value)
    model = utils.add_sampling_to_autoregressive_model(model, single_output)
  else:
    model = utils.make_model_with_empty_targets(
        model, task, _COMPUTATION_STEPS_MULT.value, single_output
    )

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=_SEED.value + 1,
      training_steps=1000,
      log_frequency=100,
      length_curriculum=curriculum,
      batch_size=_BATCH_SIZE.value,
      task=task,
      task_name=_TASK.value,
      model=model,
      model_name="",
      model_architecture=_ARCHITECTURE.value,
      loss_fn=loss_fn,
      learning_rate=1e-3,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=100,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64,
      is_autoregressive=_IS_AUTOREGRESSIVE.value,
  )

  training_worker = training.TrainingWorker(training_params, use_tqdm=True)
  _, eval_results, _ = training_worker.run()

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[_SEQUENCE_LENGTH.value + 1 :])
  print(f'Network score: {score}')


if __name__ == '__main__':
  app.run(main)
