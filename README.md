# resnet50-car-brand-categorizer

Car Brand Categorizer is a machine learning application built to predict the brand of a car from an input image. The application utilizes a pre-trained ResNet50 model and provides a user-friendly GUI for easy interaction.

![Screenshot](https://github.com/burakkeless/resnet50-car-brand-categorizer/assets/56159332/7b375636-8839-4592-8187-4cd3f9ceeb12)

## Features

- **Machine Learning Model**: The application uses a ResNet50 model pre-trained on the ImageNet dataset to classify car images into different brands.
- **GUI Interface**: Users can easily select an image file and a pre-trained model file using a graphical interface. The application then predicts the brand of the car in the selected image.

## Usage

1. Run the GUI application:
    ```bash
    python UI.py
    ```

2. Click on the "Create machine learning model with ResNet50" button to train the ResNet50 model on your dataset (optional).
3. Click on the "Browse image" button to select an image file containing a car.
4. Click on the "Browse model" button to select a pre-trained model file (.h5 format).
5. Click on the "Find the brand!" button to predict the brand of the car in the selected image.

## Acknowledgments

- This project was inspired by the need to quickly categorize car images for various purposes.
- Special thanks to the developers of the ResNet50 model and the contributors to the TensorFlow and tkinter libraries.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
