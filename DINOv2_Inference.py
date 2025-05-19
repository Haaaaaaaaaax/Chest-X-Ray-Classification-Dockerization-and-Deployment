def predict(image_path):
    # Load model directly
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained("Haaaaaaaaaax/dinov2-Base-finetuned-chest_xray")
    model = AutoModelForImageClassification.from_pretrained("Haaaaaaaaaax/dinov2-Base-finetuned-chest_xray")

    # Load an image
    image = Image.open(image_path)  # Ensure it's in RGB format

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt") # return pythorch tensor

    # Perform inference
    with torch.no_grad(): #stop training
        outputs = model(**inputs)

    # Get the predicted label
    predictions = torch.nn.functional.sigmoid(outputs.logits)
    predicted_class = predictions.argmax().item()

    # Map the predicted class index to a label
    labels = model.config.id2label
    predicted_label = labels[predicted_class]

    return predicted_label, predictions[0][predicted_class]


if __name__ == "__main__":
    image_path = r"Data\All_Data\PNEUMONIA\person1_virus_7.jpeg"  # Replace with your image path
    predicted_label, confidence = predict(image_path)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.4f}")
