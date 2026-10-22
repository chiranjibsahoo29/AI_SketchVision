import tensorflow as tf
import json




model = tf.keras.models.load_model(
    "models/sketch_classifier.keras"
)



with open(
    "models/class_names.json",
    "r"
) as f:

    class_names = json.load(f)




print("Model loaded successfully!")

print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        i,
        "->",
        name
    )


print("\nModel input shape:")
print(model.input_shape)

print("\nModel output shape:")
print(model.output_shape)