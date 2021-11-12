"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
"""
from toolbox import *

import argparse
import random

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def run_cnn32():
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32)
        for i, samples in enumerate(samples_space):
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            cnn32 = SimpleCNN32Filter(len(classes))
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            cohen_kappa, ece, train_time, test_time = run_dn_image_set(
                cnn32,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            cnn32_kappa.append(cohen_kappa)
            cnn32_ece.append(ece)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

    print("cnn32 finished")
    write_result(prefix + "cnn32_kappa" + suffix, cnn32_kappa)
    write_result(prefix + "cnn32_ece" + suffix, cnn32_ece)
    write_result(prefix + "cnn32_train_time" + suffix, cnn32_train_time)
    write_result(prefix + "cnn32_test_time" + suffix, cnn32_test_time)


def run_cnn32_2l():
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32_2l)
        for i, samples in enumerate(samples_space):
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            cohen_kappa, ece, train_time, test_time = run_dn_image_set(
                cnn32_2l,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            cnn32_2l_kappa.append(cohen_kappa)
            cnn32_2l_ece.append(ece)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l_kappa" + suffix, cnn32_2l_kappa)
    write_result(prefix + "cnn32_2l_ece" + suffix, cnn32_2l_ece)
    write_result(prefix + "cnn32_2l_train_time" + suffix, cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time" + suffix, cnn32_2l_test_time)


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_train_time = []
    cnn32_5l_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32_5l)
        for i, samples in enumerate(samples_space):
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            cohen_kappa, ece, train_time, test_time = run_dn_image_set(
                cnn32_5l,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            cnn32_5l_kappa.append(cohen_kappa)
            cnn32_5l_ece.append(ece)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l_kappa" + suffix, cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece" + suffix, cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_train_time" + suffix, cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time" + suffix, cnn32_5l_test_time)


def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (resnet18)
        for i, samples in enumerate(samples_space):
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            res = models.resnet18(pretrained=True)
            num_ftrs = res.fc.in_features
            res.fc = nn.Linear(num_ftrs, len(classes))
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            cohen_kappa, ece, train_time, test_time = run_dn_image_set(
                res,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            resnet18_kappa.append(cohen_kappa)
            resnet18_ece.append(ece)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

    print("resnet18 finished")
    write_result(prefix + "resnet18_kappa" + suffix, resnet18_kappa)
    write_result(prefix + "resnet18_ece" + suffix, resnet18_ece)
    write_result(prefix + "resnet18_train_time" + suffix, resnet18_train_time)
    write_result(prefix + "resnet18_test_time" + suffix, resnet18_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Example usage: python cifar_10.py -m 3 -s l
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    parser.add_argument("-s", help="computation speed")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"
    samples_space = np.geomspace(10, 10000, num=8, dtype=int)

    nums = list(range(10))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    if args.s == "h":
        # High speed RF
        rf_times = produce_mean(load_result(prefix + "naive_rf_train_time.txt"))
        suffix = "_st.txt"
        ratio = 1.0
    elif args.s == "l":
        # Low speed RF
        rf_times = produce_mean(load_result(prefix + "naive_rf_train_time_lc.txt"))
        suffix = "_sc.txt"
        ratio = 0.11 / 0.9
    else:
        raise Exception("Wrong configurations for time calibration.")

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    run_cnn32()
    run_cnn32_2l()
    run_cnn32_5l()

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    run_resnet18()
