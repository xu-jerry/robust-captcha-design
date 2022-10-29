


image = load_image('archive/_test/airplane/airplane_0582.jpg')

model = load_model('my_model.h5')
classifier = KerasClassifier(model=model, clip_values=(0, 255)

attacker = FastGradientMethod(eps=8)
adversarial_image = attacker.generate(image)

filtered_adversarial_image = FeatureSqueezing(bit_depth=3)(adversarial_image, clip_values=(0, 255))

prediction = classifier.predict(filtered_adversarial_image)