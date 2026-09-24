import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.data.dataset import load_medmnist_splits
from client.model.cnn import LocalClientCNN
from client.evaluate import evaluate_client_model

def test_metrics_and_evaluation():
    device = torch.device("cpu")
    _, _, test_ds = load_medmnist_splits()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    model = LocalClientCNN(in_channels=1, num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    metrics = evaluate_client_model(model, test_loader, criterion, device)

    print("\n--- Evaluation Metrics Engine Test Results ---")
    print(f"Test Loss:        {metrics['loss']:.4f}")
    print(f"Test Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall (Sens.):   {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1_score']:.4f}")
    print(f"Specificity:      {metrics['specificity']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix'].numpy()}")

    assert 0.0 <= metrics['accuracy'] <= 1.0, "Accuracy out of bounds."
    assert 0.0 <= metrics['f1_score'] <= 1.0, "F1 score out of bounds."
    print("\n✅ All Metrics & Client Evaluation Unit Tests Passed!")

if __name__ == "__main__":
    test_metrics_and_evaluation()