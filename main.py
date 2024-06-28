import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pprint import pprint

from VGG16 import ConvNetVGG16
from CNN import ConvNet


def main():
    # config
    num_epochs = 30
    batch_size = 4
    learning_rate = 1e-3
    save_model_every = 3 # epochs
    load_epoch = 0
    device = torch.device("cpu")
    train_model = True
    eval_model = True
    MODEL_NAME = "cnn"
    MODEL_OUTPUT_PATH = "cnn_logs"
    model = ConvNet()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    print("loading data...")

    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )

    classes = train_data.classes

    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.mkdir(MODEL_OUTPUT_PATH)

    print("loading model...")
    # load saved model
    if load_epoch:
        model.load_state_dict(torch.load(f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{load_epoch}.pth"))
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if train_model:
        print("initialising training...")
        # TRAIN MODEL
        for epoch in range(load_epoch, load_epoch+num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # forward pass
                outputs = model(images)
                loss = loss_func(outputs, labels)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 1000 == 0:
                    print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.4f}")

            if save_model_every and epoch % save_model_every == 0:
                torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{epoch}.pth")

        print("training complete...")
        # save output model
        torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}.pth")

    if eval_model:
        # EVALUATE MODEL
        print("evaluating model...")
        all_preds = np.array([])
        all_labels = np.array([])

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                outputs = model(images)

                _, predictions = torch.max(outputs, 1)

                all_preds = np.append(all_preds, predictions)
                all_labels = np.append(all_labels, labels)

        scores_per_class = {}
        for i, class_ in enumerate(classes):
            tp = ((all_preds == i) & (all_labels == i)).sum().item()
            tn = ((all_preds != i) & (all_labels != i)).sum().item()
            fp = ((all_preds == i) & (all_labels != i)).sum().item()
            fn = ((all_preds != i) & (all_labels == i)).sum().item()
            scores_per_class[class_] = {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Recall": tp / (tp + fn) if tp + fn else 0,
                "Precision": tp / (tp + fp) if tp + fp else 0,
                "F1-Score": tp / (tp + 0.5 * (fp + fn)) if tp + 0.5 * (fp + fn) else 0,
            }

        scores_df = pd.DataFrame(scores_per_class).T.reset_index(names="class")

        scores_df = pd.concat(
            [
                scores_df,
                pd.DataFrame.from_dict(
                    {
                        "class": "Overall",
                        "TP": scores_df["TP"].sum(),
                        "TN": scores_df["TN"].sum(),
                        "FP": scores_df["FP"].sum(),
                        "FN": scores_df["FN"].sum(),
                        "Recall": scores_df["Recall"].mean(),
                        "Precision": scores_df["Precision"].mean(),
                        "F1-Score": scores_df["F1-Score"].mean(),
                    },
                    orient="index",
                ).T,
            ]
        ).reset_index(drop=True)

        print("=" * 50)
        print("Evaluation scores")
        print("=" * 50)

        pprint(scores_df)


if __name__ == "__main__":
    main()
