# Original code: Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified code: Copyright (c) 2025.
# Licensed under the MIT License. See LICENSE for details.

from __future__ import annotations

import math
import warnings
from enum import Enum
from functools import wraps
import torch
from fsspec.utils import nullcontext
from tensordict import TensorDict, TensorDictBase, TensorDictParams

from tensordict.nn import (
    dispatch,
    set_skip_existing,
    TensorDictModule,
    TensorDictModuleBase,
)
from tensordict.utils import expand_right, NestedKey
from torch import Tensor, device
from torchrl.data.tensor_specs import Composite, TensorSpec
from torchrl.data.utils import _find_action_space
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives import LossModule, ValueEstimators, SoftUpdate
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    _vmap_func,
    default_value_kwargs,
    distance_loss,
    ValueEstimators, hold_out_net,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator, ValueEstimatorBase
from torchrl.objectives.value.advantages import _self_set_skip_existing, _self_set_grad_enabled
from torchrl.objectives.value.functional import SHAPE_ERR
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type, Union, List
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
try:
    from torch import vmap
except ImportError as err:
    try:
        from functorch import vmap
    except ImportError:
        raise ImportError(
            "vmap couldn't be found. Make sure you have torch>2.0 installed."
        ) from err

torch.autograd.set_detect_anomaly(True)
class MOCEstimator(ValueEstimatorBase):
    """​​Martingale Orthogonality Conditions Policy Evaluation.

    Keyword Args:
        gamma (scalar): discount factor.
        value_network (TensorDictModule): value operator used to retrieve
            the value estimates.
        shifted (bool, optional): if ``True``, the value and next value are
            estimated with a single call to the value network. This is faster
            but is only valid whenever (1) the ``"next"`` value is shifted by
            only one time step (which is not the case with multi-step value
            estimation, for instance) and (2) when the parameters used at time
            ``t`` and ``t+1`` are identical (which is not the case when target
            parameters are to be used). Defaults to ``False``.
        average_rewards (bool, optional): if ``True``, rewards will be standardized
            before target is computed.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, i.e., the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        device (torch.device, optional): the device where the buffers will be instantiated.
            Defaults to ``torch.get_default_device()``.

    """

    def __init__(
            self,
            *,
            gamma: float | torch.Tensor,
            alpha: float | torch.Tensor,
            value_network: TensorDictModule,
            shifted: bool = False,  
            average_rewards: bool = False,
            differentiable: bool = True,  
            advantage_key: NestedKey = None,
            value_target_key: NestedKey = None,
            value_key: NestedKey = None,
            skip_existing: bool | None = None,
            device: torch.device | None = None,
            target_entropy: Union[str, float] = "auto",
            dt: float,  
    ):
        super().__init__(
            value_network=value_network,
            differentiable=differentiable,
            shifted=shifted,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
        )
        self.dt = dt
        self.device = device
        self.register_buffer("gamma", torch.tensor(gamma, device=self.device))

        if isinstance(gamma, (float, int)):
            gamma = torch.tensor(gamma, device=self.device)
        self.register_buffer("gamma", gamma.to(self.device))

        if isinstance(alpha, (float, int)):
            alpha = torch.tensor(alpha, device=self.device)
        self.register_buffer("alpha", alpha.to(self.device))

        self.average_rewards = average_rewards
        #Check if the current object already has a target_entropy attribute. 
        #If not, proceed with the subsequent registration operation.
        #auto: Automatically compute based on the action space
        if target_entropy == "auto":
            target_entropy = -1.0  
        if not hasattr(self, "target_entropy"):
            self.register_buffer("target_entropy", torch.tensor(float(target_entropy), device=self.device))

        if not hasattr(self, "alpha"):
            self.register_buffer("alpha", torch.tensor(float(alpha), device=self.device))

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
            self,
            tensordict: TensorDictBase,
            *,
            params: TensorDictBase | None = None,
            target_params: TensorDictBase | None = None,
    ) -> TensorDictBase:

        """        Computes the  given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``, and ``"next"``
                tensordict state as returned by the environment) necessary to
                compute the value estimates and the MOCEstimate.
                The data passed to this module should be structured as
                :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the
                feature dimension(s). The tensordict must have shape ``[*B, T]``.

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too.
        """

        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )

        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )

        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                    params is None and target_params is None
            ) else nullcontext():
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        value_target = self.value_estimate(tensordict, next_value=next_value)
        tensordict.set(self.tensor_keys.value_target, value_target)
        return tensordict

    def td0_return_estimate(
            self,
            tensordict: TensorDictBase = None,
            gamma: float =None,
            next_state_value: torch.Tensor =None,
            reward: torch.Tensor =None,
            terminated: torch.Tensor | None = None,
            *,
            done: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if done is not None and terminated is None:
            terminated = done.clone()
            warnings.warn(
                "done for td0_return_estimate is deprecated. Pass ``terminated`` instead."
            )

        if reward is None:
            reward = tensordict.get(("next", self.tensor_keys.reward))
        if terminated is None:
            terminated = tensordict.get(("next", self.tensor_keys.terminated))
        if next_state_value is None:
            next_state_value = tensordict.get(("next", self.tensor_keys.value))
        if gamma is not None:
            self.gamma = gamma

        if not (next_state_value.shape == reward.shape == terminated.shape):
            raise RuntimeError(SHAPE_ERR)
        not_terminated = (~terminated).int()

        alpha = self.alpha.to(reward.device)
        target_entropy = self.target_entropy.to(reward.device)
       
        #target_value= reward + not_terminated * next_state_value
        target_value= reward*self.dt + not_terminated * next_state_value
        
        return target_value

    def value_estimate(
            self,
            tensordict: TensorDictBase,
            target_params: TensorDictBase | None = None,
            next_value: torch.Tensor | None = None,
            **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device

        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-5)
            tensordict.set(
                ("next", self.tensor_keys.reward), reward
            )  


        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        terminated = tensordict.get(("next", self.tensor_keys.terminated))
        done = tensordict.get(("next", self.tensor_keys.done), default=None)
        if terminated is None and done is not None:
            terminated = done.clone()
            warnings.warn(
                "done for td0_return_estimate is deprecated. Pass ``terminated`` instead."
            )

        value_target = self.td0_return_estimate(
            tensordict=tensordict,
            gamma=gamma,
            next_state_value=next_value,
            reward=reward,
            terminated=terminated,
            done=done,
        )
        return value_target

    def _call_value_nets(
        self,
        data: TensorDictBase,
        params: TensorDictBase | None = None,
        next_params: TensorDictBase | None = None,
        single_call: bool = False,
        value_key: NestedKey | None = None,
        detach_next: bool = True,#False,
        vmap_randomness: str | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Call the value network to obtain values of the current state and the next state.
        if value_key is None:
            value_key = self.tensor_keys.value

        if single_call:
            with self.value_network_params.to_module(self.value_network):
                data = self.value_network(data)
            value = data.get(value_key)
            next_value = data.get(("next", value_key))
            if detach_next:
                next_value = next_value.detach()
        else:
            with self.value_network_params.to_module(self.value_network):
                current_data = data.select(*self.value_network.in_keys, strict=False)
                self.value_network(current_data)
                value = current_data.get(value_key)

            with self.target_value_network_params.to_module(self.value_network):
                # noinspection PyArgumentList
                next_data = data.get("next").select(*self.value_network.in_keys, strict=False)
                self.value_network(next_data)
                next_value = next_data.get(value_key)
                if detach_next:
                    next_value = next_value.detach()
        return value, next_value

def _delezify(func):
    @wraps(func)
    def new_func(self, *args, **kwargs):
        self.target_entropy
        return func(self, *args, **kwargs)
    return new_func
def compute_log_prob(action_dist, action_or_tensordict, tensor_key):
    """Compute the log probability of an action given a distribution."""
    if isinstance(action_or_tensordict, torch.Tensor):
        log_p = action_dist.log_prob(action_or_tensordict)
    else:
        maybe_log_prob = action_dist.log_prob(action_or_tensordict)
        if not isinstance(maybe_log_prob, torch.Tensor):
            log_p = maybe_log_prob.get(tensor_key)
        else:
            log_p = maybe_log_prob
    return log_p

class ValueEstimators(Enum):
    TD0 = "Bootstrapped TD (1-step return)"
    TD1 = "TD(1) (infinity-step return)"
    TDLambda = "TD(lambda)"
    GAE = "Generalized advantage estimate"
    VTrace = "V-trace"
    MOC = "Martingale orthogonality condition"

class TestLoss(LossModule):
    """TorchRL implementation of the Test loss.

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        value_network (TensorDictModule): V(s) parametric model.
            This module typically outputs a ``"state_value"`` entry.
            If a single instance of `qvalue_network` is provided, it will be duplicated ``num_value_nets``
            times. If a list of modules is passed, their
            parameters will be stacked unless they share the same identity (in which case
            the original parameter will be expanded).

            .. warning:: When a list of parameters if passed, it will __not__ be compared against the policy parameters
              and all the parameters will be considered as untied.

    Keyword Args:
        num_value_nets (integer, optional): number of Value networks used.
            Defaults to ``2``.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        alpha_init (:obj:`float`, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (:obj:`float`, optional): min value of alpha.
            Default is None (no minimum value).
        max_alpha (:obj:`float`, optional): max value of alpha.
            Default is None (no maximum value).
        action_spec (TensorSpec, optional): the action tensor spec. If not provided
            and the target entropy is ``"auto"``, it will be retrieved from
            the actor.
        fixed_alpha (bool, optional): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            Default is ``False``.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        delay_actor (bool, optional): Whether to separate the target actor
            networks from the actor networks used for data collection.
            Default is ``False``.
        delay_value (bool, optional): Whether to separate the target value
            networks from the value networks used for data collection.
            Default is ``True``.
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            Tensordict key where to write the
            priority (for prioritized replay buffer usage). Defaults to ``"td_error"``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        skip_done_states (bool, optional): whether the actor network used for value computation should only be run on
            valid, non-terminating next states. If ``True``, it is assumed that the done state can be broadcast to the
            shape of the data and that masking the data results in a valid data structure. Among other things, this may
            not be true in MARL settings or when using RNNs. Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.test import TestLoss
        >>> from tensordict import TensorDict
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor = ProbabilisticActor(
        ...     module=module,
        ...     in_keys=["loc", "scale"],
        ...     spec=spec,
        ...     distribution_class=TanhNormal)
        >>> class ValueClass(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(n_obs , 1)
        ...     def forward(self, obs):
        ...         return self.linear(obs)
        >>> module = ValueClass()
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=['observation'])
        >>> loss = TestLoss(actor, value)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> data = TensorDict({
        ...         "observation": torch.randn(*batch, n_obs),
        ...         "action": action,
        ...         ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...         ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...         ("next", "reward"): torch.randn(*batch, 1),
        ...         ("next", "observation"): torch.randn(*batch, n_obs),
        ...     }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor, value network.
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_value", "loss_alpha", "alpha", "entropy"]``

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.test import TestLoss
        >>> _ = torch.manual_seed(42)
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor = ProbabilisticActor(
        ...     module=module,
        ...     in_keys=["loc", "scale"],
        ...     spec=spec,
        ...     distribution_class=TanhNormal)
        >>> class ValueClass(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(n_obs, 1)
        ...     def forward(self, obs):
        ...         return self.linear(obs)
        >>> module = ValueClass()
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation"])
        >>> loss = TestLoss(actor, value)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> loss_actor, loss_value, _, _, _, _ = loss(
        ...     observation=torch.randn(*batch, n_obs),
        ...     action=action,
        ...     next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_observation=torch.zeros(*batch, n_obs),
        ...     next_reward=torch.randn(*batch, 1))
        >>> loss_actor.backward()

    Examples:
        >>> _ = loss.select_out_keys('loss_actor', 'loss_value')
        >>> loss_actor, loss_value = loss(
        ...     observation=torch.randn(*batch, n_obs),
        ...     action=action,
        ...     next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_observation=torch.zeros(*batch, n_obs),
        ...     next_reward=torch.randn(*batch, 1))
        >>> loss_actor.backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"advantage"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected.  Defaults to ``"state_action_value"``.
            log_prob (NestedKey): The input tensordict key where the log probability is expected.
                Defaults to ``"sample_log_prob"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "value"
        log_prob: NestedKey = "sample_log_prob"
        priority: NestedKey = "td_error" # martingale td error
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.MOC 

    actor_network: TensorDictModule
    value_network: TensorDictModule
    actor_network_params: TensorDictParams
    value_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams | None
    target_value_network_params: TensorDictParams | None

    def __init__(
            self,
            actor_network: ProbabilisticActor,
            value_network: Optional[TensorDictModule | List[TensorDictModule]] = None,
            value_network_params: Optional[TensorDictParams | List[TensorDictParams]] = None,
            *,
            loss_function: str = MISSING, #"l2", 
            alpha_init: float = MISSING, # 1.0,
            min_alpha: float = None,
            max_alpha: float = None,
            action_spec=None,
            fixed_alpha: bool = False,
            target_entropy: Union[str, float] =  MISSING, #"auto",
            gamma: float = MISSING, 
            delay_actor: bool = False, #True, 
            delay_value: bool = True,
            priority_key: str = None,
            separate_losses: bool = True, 
            reduction: str = None,
            skip_done_states: bool = False,
            dt: float,  
    ) -> None:
        self.dt=dt
        #beta=-math.log(gamma)/delta t,delta t=0.1
        self.gamma=-math.log(gamma)
        self._in_keys = None
        self._out_keys = None
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._set_deprecated_ctor_keys(priority_key=priority_key)

        self.delay_actor = delay_actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )

        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
            value_policy_params = None

        #Is it necessary to set up a target network for stability
        self.delay_value = delay_value
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
            compare_against=policy_params,
        )

        self.loss_function = loss_function
        # Same with torchrl.objectives.sac
        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = getattr(torch, "get_default_device", lambda: torch.device("cpu"))()
        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        if bool(min_alpha) ^ bool(max_alpha):
            min_alpha = min_alpha if min_alpha else 0.0
            if max_alpha == 0:
                raise ValueError("max_alpha must be either None or greater than 0.")
            max_alpha = max_alpha if max_alpha else 1e9
        if min_alpha:
            self.register_buffer(
                "min_log_alpha", torch.tensor(min_alpha, device=device).log()
            )
        else:
            self.min_log_alpha = None
        if max_alpha:
            self.register_buffer(
                "max_log_alpha", torch.tensor(max_alpha, device=device).log()
            )
        else:
            self.max_log_alpha = None
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        self._target_entropy = target_entropy
        self._action_spec = action_spec

        self.__dict__["actor_critic"] = ActorCriticWrapper(
            self.actor_network, self.value_network
        )

        self.reduction = reduction
        self.skip_done_states = skip_done_states

#If using multiple Q or V networks, perform parallel computation across the networks
  #  def _make_vmap(self):
  #      self._vmap_networkN0 = _vmap_func(
  #          self.value_network, (None,0), randomness=self.vmap_randomness
  #      )
  #      self._vmap_network00 = _vmap_func(
  #          self.value_network,(0,None), randomness=self.vmap_randomness
  #      )

# Same with torchrl.objectives.sac
    @property
    def target_entropy_buffer(self):
        return self.target_entropy

    @property
    def target_entropy(self):
        target_entropy = self._buffers.get("_target_entropy", None)
        if target_entropy is not None:
            return target_entropy
        target_entropy = self._target_entropy
        action_spec = self._action_spec
        actor_network = self.actor_network
        device = next(self.parameters()).device
        if target_entropy == "auto":
            action_spec = (
                action_spec
                if action_spec is not None
                else getattr(actor_network, "spec", None)
            )
            if action_spec is None:
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitely or provide the spec of the "
                    "action tensor in the actor network."
                )
            if not isinstance(action_spec, Composite):
                action_spec = Composite({self.tensor_keys.action: action_spec})
            if (
                    isinstance(self.tensor_keys.action, tuple)
                    and len(self.tensor_keys.action) > 1
            ):
                action_container_shape = action_spec[self.tensor_keys.action[:-1]].shape
            else:
                action_container_shape = action_spec.shape
            target_entropy = -float(
                action_spec.shape[len(action_container_shape):].numel()
            )
        delattr(self, "_target_entropy")
        self.register_buffer(
            "_target_entropy", torch.tensor(target_entropy, device=device)
        )
        return self._target_entropy

    state_dict = _delezify(LossModule.state_dict)
    load_state_dict = _delezify(LossModule.load_state_dict)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )
        self._set_in_keys()

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        self.value_net = self.actor_critic
        alpha = self._alpha
        #hp = dict(default_value_kwargs(value_type))
        # Directly provide the hyperparameters required by MOCEstimator instead of 
        # relying on default_value_kwargs  
        hp = {
            'gamma': hyperparams.get('gamma', self.gamma),
            'alpha': hyperparams.get('alpha', alpha),
            "differentiable": True,
        }
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=self.value_net,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=self.value_net,
            )
        elif value_type is ValueEstimators.MOC:
            self._value_estimator = MOCEstimator(
                **hp,
                value_network=self.value_net,
                device=self.device,
                dt=self.dt
            )

        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=self.value_net,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        # Set keys to facilitate MOCEstimator's access and storage  
        tensor_keys = {
            "value_target": "value_target",
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        raise RuntimeError(
            "At least one of the networks of SACLoss must have trainable " "parameters."
        )

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
        ]
        keys.extend(self.value_network.in_keys)
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_actor", "loss_value", "loss_alpha", "alpha", "entropy"]
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    #Loss
    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        loss_value, value_metadata = self._value_loss(tensordict)
        # Store the Martingale td error in the data dictionary.
        tensordict.set(self.tensor_keys.priority, value_metadata["td_error"])
        loss_actor, metadata_actor = self._actor_loss(tensordict)
        loss_alpha = self._alpha_loss(log_prob=metadata_actor["log_prob"])
        if (loss_value is not None and loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"actor and value Losses shape mismatch: {loss_actor.shape} and {loss_value.shape}"
            )
        entropy = -metadata_actor["log_prob"]
        out = {
            "loss_actor": loss_actor,
            "loss_value": loss_value,
            "loss_alpha": loss_alpha,
            "alpha": self._alpha,
            "entropy": entropy.detach().mean(),
        }
        td_out = TensorDict(out, [])
        #Apply the _reduce function to each "loss_" key-value pair in td_out to perform 
        #a dimensionality reduction operation (averaging).
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

    @property
    @_cache_values
    # Detach target network to avoid gradient updates  
    def _cached_detached_value_params(self):
            return self.value_network_params.detach()

    def _actor_loss(
            self, tensordict: TensorDictBase,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:

        td_a = tensordict.clone(False)
        
        # Use set_exploration_type(ExplorationType.RANDOM) to set the exploration type to random  
        # Ensure the actor network can sample actions randomly during training. Load the Actor
        # network's parameters into the current computation graph.  
        with set_exploration_type(
                ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
        # Obtain the action distribution (dist) from the given states and use rsample()  
        # to perform reparameterized sampling, enabling gradient-based optimization.  
            dist = self.actor_network.get_dist(td_a)
            a_reparm = dist.rsample()
        log_prob = compute_log_prob(dist, a_reparm, self.tensor_keys.log_prob)

        alpha=self._alpha
        td_error = td_a.get(self.tensor_keys.priority)
        td_error = td_error.detach()  
        
        #dt=0.01
        #loss_actor=(log_prob *td_error) - alpha*0.1*self.dt* log_prob
        #dt=0.01,scale
        loss_actor=(log_prob *td_error) - 10*alpha*0.1*self.dt* log_prob
        return loss_actor, {"log_prob":log_prob.detach()}

    @property
    @_cache_values
    # Cache parameters of the target Actor and target Value networks for soft updates  
    def _cached_target_params_actor_value(self):
        return TensorDict._new_unsafe(
            {
                "module": {
                    "0": self.target_actor_network_params,
                    "1": self.target_value_network_params,
                }
            },
            torch.Size([]),
        )

    def _value_loss(
            self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # value loss
        tensordict = tensordict.clone(False)

        # Select keys relevant to the value network's inputs from the TensorDict, 
        # and create a new td_copy object  
        td_copy = tensordict.select(*self.value_network.in_keys, strict=False).detach()

        with self.value_network_params.to_module(self.value_network):
            td_copy =self.value_network(td_copy)

        with self.target_actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(td_copy)

        action = tensordict.get(self.tensor_keys.action)

        log_p = compute_log_prob(dist, action, self.tensor_keys.log_prob).detach()

        alpha=self._alpha

        pred_val =  td_copy.get(self.tensor_keys.value).squeeze(-1)
        #t=0.01
        pred_val=pred_val*10

        #t=0.01
        target_val = self.value_estimator.value_estimate(tensordict).squeeze(-1)- alpha * log_p * 0.1*self.dt
        target_val = target_val *10
        
        
        loss_value = distance_loss(
          (1 + self.gamma*self.dt ) *pred_val, target_val, loss_function=self.loss_function
        )
    
        metadata = {"td_error": ((1 + self.gamma*self.dt ) *pred_val-target_val).detach()}

        return loss_value, metadata

    def _alpha_loss(self, log_prob: Tensor) -> Tensor:
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_prob + self.target_entropy)
            
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_prob)
        return alpha_loss
    @property
    def _alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

class Test(Algorithm):
    """Continuous Multi Agent Soft Actor Critic.

    Args:
        share_param_critic (bool): Whether to share the parameters of the critics withing agent groups
        num_value_nets (integer): number of Value networks used.
        loss_function (str): loss function to be used with
            the value function loss.
        delay_value (bool): Whether to separate the target value
            networks from the value networks used for data collection.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        alpha_init (float): initial entropy multiplier.
        min_alpha (float): min value of alpha.
        max_alpha (float): max value of alpha.
        fixed_alpha (bool): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
        scale_mapping (str): positive mapping function to be used with the std.
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        use_tanh_normal (bool): if ``True``, use TanhNormal as the continuyous action distribution with support bound
            to the action domain. Otherwise, an IndependentNormal is used.
        gamma (float):discount factor
    """

    def __init__(
            self,
            share_param_critic: bool,
            num_value_nets: int,
            loss_function: str,
            target_entropy: Union[float, str],
            alpha_init: float,
            min_alpha: Optional[float],
            max_alpha: Optional[float],
            fixed_alpha: bool,
            scale_mapping: str,
            use_tanh_normal: bool,
            gamma: float,
            dt: float,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dt= dt
        self.share_param_critic = share_param_critic
        self.num_value_nets = num_value_nets
        self.loss_function = loss_function
        self.target_entropy = target_entropy
        self.alpha_init = alpha_init
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.fixed_alpha = fixed_alpha
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.gamma = gamma
    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
            self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # Loss
        loss_module = TestLoss(
            actor_network=policy_for_loss,
            value_network=self.get_continuous_value_module(group),
            loss_function=self.loss_function,
            alpha_init=self.alpha_init,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            action_spec=self.action_spec,
            fixed_alpha=self.fixed_alpha,
            target_entropy=self.target_entropy,
            gamma=self.gamma,
            dt=self.dt
        )
        loss_module.set_keys(
            value=(group, "value"),
            action=(group, "action"),
            reward=(group, "reward"),
            priority=(group, "td_error"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )

        loss_module.make_value_estimator(
            ValueEstimators.MOC, gamma=self.experiment_config.gamma  
        )
        return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        items = {
            "loss_actor": list(loss.actor_network_params.flatten_keys().values()),
            "loss_value": list(loss.value_network_params.flatten_keys().values()),
        }
        if not self.fixed_alpha:
            items.update({"loss_alpha": [loss.log_alpha]})
        return items

    def _get_policy_for_loss(
            self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False, 
            share_params=self.share_param_critic,  
            device=self.device,
            action_spec=self.action_spec,
        )

        extractor_module = TensorDictModule(
            NormalParamExtractor(scale_mapping=self.scale_mapping),
            in_keys=[(group, "logits")],
            out_keys=[(group, "loc"), (group, "scale")],
        )
        policy = ProbabilisticActor(
            module=TensorDictSequential(actor_module, extractor_module),
            spec=self.action_spec[group, "action"],
            in_keys=[(group, "loc"), (group, "scale")],
            out_keys=[(group, "action")],
            distribution_class=(
                IndependentNormal if not self.use_tanh_normal else TanhNormal
            ),
            distribution_kwargs=(
                {
                    "low": self.action_spec[(group, "action")].space.low,
                    "high": self.action_spec[(group, "action")].space.high,
                }
                if self.use_tanh_normal
                else {}
            ),
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )
        return policy

    def _get_policy_for_collection(
            self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        return batch

    #####################
    # Custom new methods
    #####################

    def get_continuous_value_module(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        modules = []

        if self.share_param_critic:
            critic_output_spec = Composite(
                {"value": Unbounded(shape=(1,))}
            )
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:

            modules.append(
                TensorDictModule(
                    lambda action: action.reshape(*action.shape[:-2], -1),
                    in_keys=[(group, "action")],
                    out_keys=["global_action"],
                )
            )

            critic_input_spec = self.state_spec.clone().update(
                {
                    "global_action": Unbounded(
                        shape=(self.action_spec[group, "action"].shape[-1] * n_agents,)
                    )
                }
            )

            modules.append(
                self.critic_model_config.get_model(
                    input_spec=critic_input_spec,
                    output_spec=critic_output_spec,
                    n_agents=n_agents,
                    centralised=True,
                    input_has_agent_dim=False,
                    agent_group=group,
                    share_params=self.share_param_critic,
                    device=self.device,
                    action_spec=self.action_spec,
                )
            )

        else:
            critic_input_spec = Composite(
                {
                    group: self.observation_spec[group]
                    .clone()
                    .update(self.action_spec[group])
                }
            )

            modules.append(
                self.critic_model_config.get_model(
                    input_spec=critic_input_spec,
                    output_spec=critic_output_spec,
                    n_agents=n_agents,
                    centralised=True,
                    input_has_agent_dim=True,
                    agent_group=group,
                    share_params=self.share_param_critic,
                    device=self.device,
                    action_spec=self.action_spec,
                )
            )

        if self.share_param_critic:
            modules.append(
                TensorDictModule(
                    lambda value: value.unsqueeze(-2).expand(
                        *value.shape[:-1], n_agents, 1
                    ),
                    in_keys=["value"],
                    out_keys=[(group, "value")],
                )
            )

        return TensorDictSequential(*modules)

@dataclass
class TestConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Test`."""

    share_param_critic: bool = MISSING
    num_value_nets: int = MISSING
    loss_function: str = MISSING
    target_entropy: Union[float, str] = MISSING
    gamma: float = MISSING
    alpha_init: float = MISSING
    min_alpha: Optional[float] = MISSING
    max_alpha: Optional[float] = MISSING
    fixed_alpha: bool = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    dt: float = MISSING 

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Test

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return False

