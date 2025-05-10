from pathlib import Path
import sys
import time
import re
import json
import threading
from itertools import product
from churn_sim import NodeSystemSimulation
from sys_runner import DockerNodeSystem
from collections import OrderedDict
import pandas as pd
import os
import shutil
from sklearn.model_selection import StratifiedKFold

# ====== Environment ======
PARTIES_HOST = 'localhost'     # parties should always be run on localhost
CLOUD_HOST = 'localhost'  # hostname or ip of the cloud docker host
EPOCH_TIME = 1                 # time resolution of the failure process simulation
START_WITH_THRESH = False # if true, starts with exactly the threshold number of nodes. Otherwise, starts with the expected number of node for the experiment's failure rate.
EXP_SKIP_THRESH = 0.2 # experiments for which the expected time above threshold is below 20% are skipped

# ====== Networking parameters ======
RATE_LIMIT = "100mbit" # outbound rate limit for the parties
DELAY = "30ms"         # outbound network delay for the parties

# ====== Experiment parameters ======
EVAL_COUNT = (
    -1
)  # Currently unused, but can be used to limit the number of rounds in the experiment if adapting Go code
N_REP = 1  # number of experiment repetition
SKIP_TO = 0  # starts from a specific experiment number in the grid

# ====== Experiment mode ======
EXP_MODE = "predictive_performance"  # "predictive_performance" or "runtime_performance"

# ====== Experiment Grid ======
N_PARTIES = [2, 5, 10, 50]  # the number of session nodes
NUMBER_ESTIMATORS = [1, 10, 100, 500]
TREE_DEPTH = [1, 3, 5]
THRESH_VALUES = [1]  # the cryptographic threshold in percentage of the number of nodes
FAILURE_RATES = [0]  # the failure rate in fail/min
FAILURE_DURATIONS = [
    0.1
]  # the mean failure duration in min, cannot be zero (zero division in the simulation)
RANDOM_SEED = 0  # Random state for the StratifiedKFold

EXPERIMENTS_FOLDER = "helium/exp_runner/data/experiments"
DATASETS_FOLDER = "helium/exp_runner/data/datasets"
DATASETS = [
    "preprocessed_Ionosphere.csv",
    "preprocessed_Haberman's Survival.csv",
    "preprocessed_Breast Cancer Wisconsin (Prognostic).csv",
    "preprocessed_LTD.csv",
    "preprocessed_Mammographic Mass.csv",
    "preprocessed_Breast Cancer Wisconsin (Diagnostic).csv",
    "preprocessed_TCGA.csv",
    "preprocessed_MAGIC Gamma Telescope.csv",
    "preprocessed_Blood Transfusion Service Center.csv",
    "preprocessed_Musk (Version 2).csv",
    "preprocessed_Breast Cancer Wisconsin (Original).csv",
    "preprocessed_Spambase.csv",
]  # list all datasets that you want to use for the experiment, in runtime mode only one dataset is accepted
# only run numerical datasets, categorical datasets are not supported yet

# ====== Predictive Performance Parameters (only for the predictive_performance mode) ======
EXPERIMENT_NAME = (
    f'pred_performance_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}'
)
EXPERIMENT_FOLDER = os.path.join(
    EXPERIMENTS_FOLDER,
    EXPERIMENT_NAME,
)  # folder where the experiment results are stored

N_SPLITS = 10  # Number of splits for the cross-validation with a StratifiedKFold
FOLD_RANDOM_SEED = 0  # Random state for the StratifiedKFold


def log(str, end="\n"):
    print(str, file=sys.stderr, end=end, flush=True)


def get_stats(container, print=False):
    stats_json = None
    p = re.compile(r'\d+\.\d+')
    for l in container.logs(stream=True):
        line = l.decode('utf-8')
        if line.startswith("STATS"):
            stats_json = json.loads(line.split()[1])
        if print:
            log("%s | %s" % (container.name, line), end="")
    if stats_json == None:
        raise Exception("Container terminated without outputting stats.")
    return OrderedDict([
        ("TimeSetup", float(stats_json["Time"]["Setup"])/1e9),
        ("SentSetup", float(stats_json["Net"]["Setup"]["DataSent"])/1e6),
        ("RecvSetup", float(stats_json["Net"]["Setup"]["DataRecv"])/1e6),
        ("TimeCompute", float(stats_json["Time"]["Compute"])/1e9),
        ("SentCompute", float(stats_json["Net"]["Compute"]["DataSent"])/1e6),
        ("RecvCompute", float(stats_json["Net"]["Compute"]["DataRecv"])/1e6),
    ])


def setup_predictive_performance_data():
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

    for dataset in DATASETS:
        if not os.path.exists(Path(DATASETS_FOLDER, dataset)):
            raise Exception(f"Dataset {dataset} not found in {DATASETS_FOLDER}")
        if not dataset.endswith(".csv"):
            raise Exception(f"Dataset {dataset} is not a csv file")
        dataset_path = os.path.join(DATASETS_FOLDER, dataset)
        dataset_df = pd.read_csv(dataset_path)

        current_dataset_folder = os.path.join(EXPERIMENT_FOLDER, dataset.split(".")[0])
        os.makedirs(current_dataset_folder, exist_ok=True)
        current_dataset_models_folder = os.path.join(current_dataset_folder, "models")
        os.makedirs(current_dataset_models_folder, exist_ok=True)

        # Create attribute domains
        # We assume that the last column is the target variable
        target_col_index = len(dataset_df.columns) - 1
        target_col = dataset_df.columns[target_col_index]
        attribute_domains = {
            index: (min(dataset_df[col]), max(dataset_df[col]))
            for index, col in enumerate(dataset_df.columns)
            if index != target_col_index
        }
        # Write attribute domains to file
        with open(
            os.path.join(current_dataset_folder, "attribute_domains.json"), "w"
        ) as f:
            json.dump(attribute_domains, f)

        # Create StratifiedKFolds of whole dataset
        folds = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=FOLD_RANDOM_SEED
        )
        fold_folder = os.path.join(current_dataset_folder, "folds")
        os.makedirs(fold_folder, exist_ok=True)
        for fold, (train_index, test_index) in enumerate(
            folds.split(dataset_df, dataset_df[target_col])
        ):
            train_df = dataset_df.iloc[train_index]
            test_df = dataset_df.iloc[test_index]

            # Write train and test data to file
            train_df.to_csv(
                os.path.join(fold_folder, f"train_fold_{fold}.csv"),
                index=False,
            )
            test_df.to_csv(
                os.path.join(fold_folder, f"test_fold_{fold}.csv"),
                index=False,
            )


log("Computing experiments...")


if EXP_MODE == "predictive_performance":
    log("Predictive performance mode")
    log("Setting up predictive performance data")
    setup_predictive_performance_data()


exps_to_run = []
for (
    n_party,
    thresh,
    mean_failure_per_min,
    mean_failure_duration,
    n_estimators,
    tree_depth,
) in product(
    N_PARTIES,
    THRESH_VALUES,
    FAILURE_RATES,
    FAILURE_DURATIONS,
    NUMBER_ESTIMATORS,
    TREE_DEPTH,
):

    thresh = round(n_party * thresh)
    sim = NodeSystemSimulation(
        n_party, mean_failure_per_min, mean_failure_duration, EPOCH_TIME
    )
    avg_online_count, frac_time_above_thresh = (
        sim.expected_online_nodes(),
        sim.expected_time_above_threshold(thresh),
    )
    if frac_time_above_thresh < EXP_SKIP_THRESH:
        continue
    exps_to_run.append(
        (
            n_party,
            thresh,
            mean_failure_per_min,
            mean_failure_duration,
            n_estimators,
            tree_depth,
        )
    )
log("%d experiments to run" % (len(exps_to_run)*N_REP))

for i, (exp, rep) in enumerate(product(exps_to_run, range(N_REP))):

    if i+1 < SKIP_TO:
        continue

    (
        n_party,
        thresh,
        mean_failure_per_min,
        mean_failure_duration,
        n_estimators,
        tree_depth,
    ) = exp

    # Create party chunks
    for dataset in DATASETS:
        party_data_folder = os.path.join(
            EXPERIMENT_FOLDER, dataset.split(".")[0], "party_data"
        )
        os.makedirs(party_data_folder, exist_ok=True)
        for fold in range(N_SPLITS):
            # make fold folder
            fold_folder = os.path.join(party_data_folder, f"fold_{fold}")
            os.makedirs(fold_folder, exist_ok=True)
            # read train data and store in chunks in party_data folder
            train_data = pd.read_csv(
                os.path.join(
                    EXPERIMENT_FOLDER,
                    dataset.split(".")[0],
                    "folds",
                    f"train_fold_{fold}.csv",
                )
            )

            data = train_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(
                drop=True
            )  # Shuffle the data
            k, m = divmod(len(data), n_party)
            data_chunks = [
                data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
                for i in range(n_party)
            ]

            # write out data chunks with party id to data/simulation/ folder
            for party_id, chunk in enumerate(data_chunks):
                chunk.to_csv(f"{fold_folder}/node-{party_id}.csv", index=False)

    log(
        "======= starting experiment N=%d T=%d F=%.2f REP=%d ======="
        % (n_party, thresh, mean_failure_per_min, rep)
    )

    for dataset_index, dataset in enumerate(DATASETS):
        for N_FOLD in range(N_SPLITS):
            # write out experiment_config
            try:
                print(f"Running experiment {N_FOLD} for dataset {dataset}")

                docker_mounted_experiments_folder = "/helium/data/experiments"
                experiment_config = {
                    "data_folder_path": os.path.join(
                        docker_mounted_experiments_folder,
                        EXPERIMENT_NAME,
                        DATASETS[dataset_index].split(".")[0],
                        "party_data",
                        f"fold_{N_FOLD}",
                    ),
                    "attribute_domains_path": os.path.join(
                        docker_mounted_experiments_folder,
                        EXPERIMENT_NAME,
                        DATASETS[dataset_index].split(".")[0],
                        "attribute_domains.json",
                    ),
                    "model_path": os.path.join(
                        docker_mounted_experiments_folder,
                        EXPERIMENT_NAME,
                        DATASETS[dataset_index].split(".")[0],
                        "models",
                        f"model_{DATASETS[dataset_index].split('.')[0]}_fold_{N_FOLD}_parties_{n_party}_estimators_{n_estimators}_depth_{tree_depth}.json",
                    ),
                }

                print("Writing out experiment config for ", N_FOLD)

                with open(
                    os.path.join(
                        EXPERIMENTS_FOLDER,
                        "experiment_config.json",
                    ),
                    "w",
                ) as f:
                    json.dump(experiment_config, f)

                    f.flush()
                    os.fsync(f.fileno())

                time.sleep(1)

                system = DockerNodeSystem(
                    n_party,
                    thresh,
                    PARTIES_HOST,
                    CLOUD_HOST,
                    RATE_LIMIT,
                    DELAY,
                    EVAL_COUNT,
                    n_estimators,
                    tree_depth,
                    N_FOLD,
                )

                churn_sim = NodeSystemSimulation(
                    n_party,
                    mean_failure_per_min,
                    mean_failure_duration,
                    EPOCH_TIME,
                    on_failure=system.kill_player,
                    on_reconnect=system.start_player,
                    initial_online=thresh if START_WITH_THRESH else None,
                )

                time.sleep(5)  # lets the thing clean

                exp_terminated = threading.Event()

                def excepthook(args):
                    time.sleep(2)
                    if not exp_terminated.is_set():
                        raise Exception(
                            "Got exception of type %s value %s during experiment"
                            % (args.exc_type, args.exc_value)
                        )

                threading.excepthook = excepthook

                cloud = system.start_cloud()

                churn_sim.run_simulation()

                stats = get_stats(cloud, print=True)
                exp_desc = OrderedDict(
                    [
                        ("threshold", thresh),
                        ("failure_rate", mean_failure_per_min),
                        ("rep", rep),
                    ]
                    + [item for item in stats.items()]
                    + [
                        ("failure_duration", mean_failure_duration),
                        ("exp", "helium"),
                        ("n_party", n_party),
                        ("rate_limit", RATE_LIMIT),
                        ("delay", DELAY),
                        ("theoretical_node_online", churn_sim.expected_online_nodes()),
                        (
                            "theoretical_time_above_thresh",
                            churn_sim.expected_time_above_threshold(thresh),
                        ),
                        ("actual_node_online", churn_sim.online_nodes()),
                        (
                            "actual_time_above_thresh",
                            churn_sim.time_above_threshold(thresh),
                        ),
                    ]
                )

                if churn_sim.failed_fail > 0 or churn_sim.failed_rec > 0:
                    print(
                        "Warning: churn simulation got %d failed failures and %d failed reconnect"
                        % (churn_sim.failed_fail, churn_sim.failed_rec)
                    )

                print(json.dumps(exp_desc), flush=True)
            except Exception as e:
                log(e)
                sys.exit(1)
            finally:
                exp_terminated.set()
                churn_sim.stop()
                system.clean_all()

            time.sleep(2)
