# Original code: Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified code: Copyright (c) 2025.
# Licensed under the MIT License. See LICENSE for details.

import torch
from pathlib import Path
from typing import List

from benchmarl.eval_results import load_and_merge_json_dicts, Plotting
from matplotlib import pyplot as plt

from benchmarl.algorithms import MasacConfig, MappoConfig, MaddpgConfig, TestConfig
from benchmarl.models.mlp import MlpConfig

def run_benchmark() -> List[str]:
    from benchmarl.algorithms import TestConfig, MaddpgConfig, IqlConfig, QmixConfig, MappoConfig, MasacConfig
    from benchmarl.benchmark import Benchmark
    from benchmarl.environments import VmasTask 
    from benchmarl.experiment import ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    # Configure experiment
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.save_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    experiment_config.loggers = []
    experiment_config.max_n_iters = 10


    # Configure benchmark
    tasks = [
        VmasTask.NAVIGATION.get_from_yaml()
        ]
    algorithm_configs = [
        QmixConfig.get_from_yaml(),
        IqlConfig.get_from_yaml(),
        MappoConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
        MaddpgConfig.get_from_yaml(),
        TestConfig.get_from_yaml(),
    ]
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds={0,1,2},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )
 
    # For each experiment, run it and get its output file name
    experiments = benchmark.get_experiments()
    experiments_json_files = []
    
    for experiment in experiments:

        exp_json_file = str(
            Path(experiment.folder_name) / Path(experiment.name + ".json")
        )
        experiments_json_files.append(exp_json_file)
        experiment.run()
    return experiments_json_files



if __name__ == "__main__":
    # Uncomment this to rerun the benchmark that generates the files
    experiments_json_files = run_benchmark()

    raw_dict = load_and_merge_json_dicts(experiments_json_files)

    # Load and process experiment outputs
    # raw_dict = load_and_merge_json_dicts(experiments_json_files)
    processed_data = Plotting.process_data(raw_dict)
    (
        environment_comparison_matrix,
        sample_efficiency_matrix,
    ) = Plotting.create_matrices(processed_data, env_name="vmas")

    # Plotting
    Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )

    Plotting.aggregate_scores(
        environment_comparison_matrix=environment_comparison_matrix,
        tabular_results_file_path=os.path.join(target_dir, "aggregated_score_return")
    )
    
    Plotting.probability_of_improvement(
        environment_comparison_matrix,
        algorithms_to_compare=[["qmix","iql","mappo","masac","maddpg"]]
    )
    plt.show()
