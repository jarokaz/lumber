import utils


images, labels = utils.load_images()

print(images[0].shape)

test_images = images[0:1]
print(len(test_images))

result=utils.classify_images(test_images)
print(result)