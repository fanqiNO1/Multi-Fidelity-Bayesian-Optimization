from turtle import ycor
import torch
import math
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch import fit_gpytorch_model

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# We will optimize the following function:
# y = x_1 ^ 2 + (2 * sin{x_2} - (1 - x_4)) * 100 + e^{x_3}
# the range of x_1 is [0, 5], x_2 is [0, 2pi], x_3 is [0, 5]
# the x_4 is fidelity, the target is 1
# the optimal value should be 25 + 2 * 100 + e^5 = 373.41

def problem(x):
    # Define the problem
    variable_num = x.shape[1]
    if variable_num != 4:
        raise ValueError("The number of variables should be 4.")
    return x[:, 0] ** 2 + (2 * torch.sin(x[:, 1]) - (1 - x[:, 3])) * 100 + torch.exp(x[:, 2])

def generate_data(n=4):
    # Generate data
    x_1 = torch.rand(n, 1, **tkwargs) * 5
    x_2 = torch.rand(n, 1, **tkwargs) * 2 * math.pi
    x_3 = torch.rand(n, 1, **tkwargs) * 5
    x_4 = torch.rand(n, 1, **tkwargs)
    x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
    y = problem(x).unsqueeze(-1)
    return x, y

def initialize_model(x, y):
    # Define a surrogate model suited for a "training data"-like fidelity parameter
    # Fidelity is in dimension 3
    # https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression_fidelity.py
    model = SingleTaskMultiFidelityGP(
        train_X=x,
        train_Y=y,
        outcome_transform=Standardize(m=1),
        data_fidelity=3
    )
    # https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/mlls/exact_marginal_log_likelihood.py
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

def initialize_bounds():
    x_1 = torch.tensor([0, 5])
    x_2 = torch.tensor([0, 2 * math.pi])
    x_3 = torch.tensor([0, 5])
    x_4 = torch.tensor([0, 1])
    bounds = torch.stack((x_1, x_2, x_3, x_4)).transpose(0, 1)
    return bounds

def initialize_target_fidelity():
    target_fidelities = {3: 1.0}
    return target_fidelities

def initialize_cost():
    # Define a cost model
    # https://github.com/pytorch/botorch/blob/main/botorch/models/cost.py
    cost_model = AffineFidelityCostModel(fidelity_weights=initialize_target_fidelity(), fixed_cost=5.0)
    # https://github.com/pytorch/botorch/blob/main/botorch/acquisition/cost_aware.py
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    return cost_model, cost_aware_utility

def project(x):
    # Project the input to the target_fidelities
    # https://github.com/pytorch/botorch/blob/main/botorch/acquisition/utils.py
    return project_to_target_fidelity(X=x, target_fidelities=initialize_target_fidelity())

def get_mfkg(model):
    # d: The feature dimension expected by acq_function.
    # columns: No idea
    # value: No idea
    # https://github.com/pytorch/botorch/blob/main/botorch/acquisition/fixed_feature.py
    current_value_acqf = FixedFeatureAcquisitionFunction(
        # https://github.com/pytorch/botorch/blob/main/botorch/acquisition/posterior_mean.py
        acq_function=PosteriorMean(model),
        d=4,
        columns=[3],
        values=[1]
    )
    # bounds: The range of the variables, fidelity is not included.
    # q: The number of candidates. Namely, batch size
    # num_restarts: The number of starting points for multistart acquisition function optimization.
    # raw_samples: The number of samples for initialization. This is required if batch_initial_conditions is not specified.
    # https://github.com/pytorch/botorch/blob/main/botorch/optim/optimize.py
    _, current_value = optimize_acqf(
        acq_function=current_value_acqf,
        bounds=initialize_bounds()[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200},
    )
    _, cost_aware_utility = initialize_cost()
    # num_fantasies: The number of fantasy points to use. More fantasypoints result in a better approximation.
    # https://github.com/pytorch/botorch/blob/main/botorch/acquisition/knowledge_gradient.py
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

# This helper function optimizes the acquisition function and returns optimal x along with the observed function values.
# If fidelity is discrete, we don't need to initialize the x.
# And we don't need to specify batch_initial_conditions in optimizae_acqf.
# But we should need to specify fidelity by fixed_features_list in optimize_acqf.
# fixed_features_list=[{fidelity_index: fidelity_1}, {fidelity_index: fidelity_2}, {fidelity_index: fidelity_3}]}],
def optimize_mfkg_and_get_observation(model, mfkg_acqf):
    # https://github.com/pytorch/botorch/blob/main/botorch/optim/initializers.py
    x_init = gen_one_shot_kg_initial_conditions(
        acq_function = mfkg_acqf,
        bounds=initialize_bounds(),
        q=4,
        num_restarts=10,
        raw_samples=512,
    )
    # batch_initial: Just the initial conditions.
    # https://github.com/pytorch/botorch/blob/main/botorch/optim/optimize.py
    candidates, _ = optimize_acqf(
        acq_function=mfkg_acqf,
        bounds=initialize_bounds(),
        q=4,
        num_restarts=10,
        raw_samples=512,
        batch_initial_conditions=x_init,
        options={"batch_limit": 5, "maxiter": 200},
    )
    cost_model, _ = initialize_cost()
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_y = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_y}\n\n")
    return new_x, new_y, cost

# Train the model by using low fidelity data.
def train():
    n_iter = 10
    cost_sum = 0
    x, y = generate_data(n=16)
    for i in range(n_iter):
        model, mll = initialize_model(x, y)
        fit_gpytorch_model(mll)
        mfkg_acqf = get_mfkg(model)
        new_x, new_y, cost = optimize_mfkg_and_get_observation(model, mfkg_acqf)
        x = torch.cat([x, new_x])
        y = torch.cat([y, new_y])
        cost_sum += cost
    return model, cost_sum

def get_result(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=4,
        columns=[3],
        values=[1],
    )
    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=initialize_bounds()[:,:-1],
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    final_rec = rec_acqf._construct_X_full(final_rec)
    objective_value = problem(final_rec)
    print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
    return final_rec

def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    model, cost = train()
    get_result(model)
    print(f"\ntotal cost: {cost}\n")
if __name__ == "__main__":
    main()