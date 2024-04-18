from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    # classifying the data
    prediction = (
        text_validation.set_index("id")["text"]
        .str.contains("in conclusion", case=False)
        .astype(int)
    )

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

    # calculating the evaluation metrics
    accuracy = accuracy_score(targets_validation["generated"], prediction["generated"])
    precision = precision_score(targets_validation["generated"], prediction["generated"])
    recall = recall_score(targets_validation["generated"], prediction["generated"])
    f1 = f1_score(targets_validation["generated"], prediction["generated"])
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

