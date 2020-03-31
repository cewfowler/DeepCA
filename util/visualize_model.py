from ann_visualizer.visualize import ann_viz;
from keras.models import model_from_json;
import argparse;


def visualize_model(model_file, weights_file):
    json_file = open(model_file, 'r');
    loaded_model_json = json_file.read();
    json_file.close();

    model = model_from_json(loaded_model_json);
    model.load_weights(weights_file)

    ann_viz(model, title="DeepCA - Model Visualization");


def main():
    parser = argparse.ArgumentParser(description="""Allows user to visualize
                model stored as a json file""");
    parser.add_argument('--model_file', help='Path to JSON model file.', required=True);
    parser.add_argument('--weights_file', help='Path to H5 weights file.', required=True);
    args = parser.parse_args();

    visualize_model(args.model_file, args.weights_file);

    return 0;

if __name__ == '__main__':
    main();
