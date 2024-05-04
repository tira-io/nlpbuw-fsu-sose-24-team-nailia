from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import BertTokenizer, BertForSequenceClassification
import torch

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    # Load BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Tokenize and encode the text data
    max_length = 128
    inputs_validation = tokenizer(text_validation['text'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # Set model to evaluation mode
    model.eval()

    # Make predictions on validation data
    with torch.no_grad():
        outputs = model(**inputs_validation)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Convert predictions to the required format
    predictions_df = text_validation[['id']].copy()
    predictions_df['generated'] = predictions.tolist()

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_df.to_json(
        Path(output_directory) / "predictions.jsonl",
        orient="records",
        lines=True,
    )
