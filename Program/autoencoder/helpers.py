import json
import numpy as np
import layers

layers_map = {
  layer.__name__: layer for layer in [
  layers.LinearLayer,
  layers.SigmoidLayer,
  layers.TanhLayer,
  layers.ReluLayer,
  layers.MSEOutputLayer]
}

def save_model(model_layers, prefix="", path="saved"):
    model_json_path = f"./{path}/{prefix}model.json"
    model_weights_path = f"./{path}/{prefix}weights.npy"
    model_json = {}
    weights = []

    with open(model_weights_path, 'wb') as weights_file:
        for idx, model_layer in enumerate(model_layers):
            if isinstance(model_layer, layers.LinearLayer):
                model_json[f"{type(model_layer).__name__}_{idx}"] = model_layer.W.shape
                np.save(weights_file, model_layer.W)
                np.save(weights_file, model_layer.b)
            else:
                model_json[f"{type(model_layer).__name__}_{idx}"] = []

        with open(model_json_path, 'w') as model_file:
            json.dump(model_json, model_file)

def load_model(prefix="", path="saved"):
    model_json_path = f"./{path}/{prefix}model.json"
    model_weights_path = f"./{path}/{prefix}weights.npy"
    model_layers = []
    
    with open(model_json_path, 'r') as model_file:
        model_json = json.load(model_file)

        
    with open(model_weights_path, 'rb') as weights_file:
        for model_layer, layer_shape in model_json.items():
            layer_name = model_layer.split('_')[0]

            if str.startswith(layer_name, layers.LinearLayer.__name__):
                weights = np.load(weights_file)
                bias = np.load(weights_file)
                linear_layer = layers.LinearLayer(layer_shape[0], layer_shape[1])
                linear_layer.W = weights.reshape(layer_shape[0], layer_shape[1])
                linear_layer.b = bias.reshape(layer_shape[1])
                model_layers.append(linear_layer)
            else:
                model_layers.append(layers_map[layer_name]())

    return model_layers
