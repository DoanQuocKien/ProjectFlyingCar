def evaluate_model(model, test_loader, label_encoder, device):
    model.eval()
    
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = correct / total
    print(f"\n✅ Test Accuracy: {acc:.4f} ({correct}/{total})")

    # Decode label về text
    pred_text = label_encoder.inverse_transform(all_preds)
    true_text = label_encoder.inverse_transform(all_labels)

    return pred_text, true_text

pred_text, true_text = evaluate_model(model, test_loader, label_encoder, device)
print(classification_report(true_text, pred_text))