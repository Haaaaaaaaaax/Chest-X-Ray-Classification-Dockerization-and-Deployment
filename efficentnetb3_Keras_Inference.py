def predict(image_path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
    from PIL import Image
    import numpy as np

    # Load your model
    model = load_model('keras_efficentnetb0_Chest_X-Ray.keras')

    # Parameters
    img_size = (300, 300)
    classes = ['NORMAL', 'PNEUMONIA']

    # Load and preprocess image function
    def load_and_preprocess_image(image_path):
        img = Image.open(image_path).convert('RGB')  # Ensure 3 channels
        img = img.resize(img_size)
        img_array = np.array(img)
        img_array = effnet_preprocess(img_array)    # Use EfficientNet preprocessing
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array, img

    # Inference
    img_array, original_img = load_and_preprocess_image(image_path)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return classes[predicted_class[0]], predictions[0][predicted_class[0]]

if __name__ == "__main__":
    image_path = r"Data\All_Data\NORMAL\IM-0137-0001.jpeg"
    predicted_label, confidence = predict(image_path)
    print(f"Predicted class: {predicted_label}, Confidence: {confidence:.4f}")