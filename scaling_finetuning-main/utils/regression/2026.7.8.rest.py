import os
import numpy as np
import matplotlib.pyplot as plt


RESULT_ROOT = "/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_original_heads_vs_fmri"


models = {
    "7B": 32,
    "13B": 40,
    "30B": 60,
}


subjects = [4,5,6]


def load_layer_r(model, layer):

    all_corr = []

    for subj in subjects:

        path = os.path.join(
            RESULT_ROOT,
            "llama",
            model,
            "base",
            f"layer{layer}",
            f"base_subj{subj}_corrlh.npy"
        )

        if not os.path.exists(path):
            print("missing:", path)
            continue

        corr = np.load(path)

        all_corr.append(corr)


    if len(all_corr)==0:
        return None


    # subject平均
    all_corr = np.array(all_corr)

    mean_subject = np.mean(all_corr, axis=0)

    return {
        "mean": np.mean(mean_subject),
        "median": np.median(mean_subject),
        "std": np.std(mean_subject)
    }



plt.figure(figsize=(10,6))


for model, n_layer in models.items():

    layer_mean = []
    layer_median = []

    layers=[]

    for layer in range(n_layer):

        result = load_layer_r(model, layer)

        if result is None:
            continue

        layers.append(layer)

        layer_mean.append(result["mean"])
        layer_median.append(result["median"])


    plt.plot(
        layers,
        layer_mean,
        marker="o",
        label=f"{model} mean"
    )


    plt.plot(
        layers,
        layer_median,
        linestyle="--",
        alpha=0.6,
        label=f"{model} median"
    )



plt.xlabel("Layer")
plt.ylabel("Pearson r")
plt.title("Layer-wise LLM-fMRI alignment")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "layer_alignment_curve.png",
    dpi=300
)

plt.show()