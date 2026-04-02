from ultralytics import YOLO
model = YOLO('best.pt')
print(model.names)  # shows exactly what IDs and names the model knows

# Test on a static image you know has people/cars in it
results = model('https://ultralytics.com/images/bus.jpg', conf=0.1)
results[0].show()