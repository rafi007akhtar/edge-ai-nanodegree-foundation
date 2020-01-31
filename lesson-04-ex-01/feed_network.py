import argparse
### TODO: Load the necessary libraries
import os
from openvino.inference_engine import IECore, IENetwork

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    ### TODO: Load the Inference Engine API
    iecore = IECore()

    ### TODO: Load IR files into their related class
    model_bin = str(os.path.splitext(model_xml)[0]) + '.bin'
    net = IENetwork(model=model_xml, weights=model_bin)

    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    iecore.add_extension(extension_path=CPU_EXTENSION, device_name="CPU")

    ### TODO: Get the supported layers of the network
    layers_map = iecore.query_network(network=net, device_name="CPU")
    print(net.layers.keys())

    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.
    unsupported_layers = [layer for layer in net.layers.keys() if layer not in layers_map]
    if len(unsupported_layers) != 0:
        print("Please add the extensions to the IECore for the following unsupported layers: ", unsupported_layers)
        exit(1)

    ### TODO: Load the network into the Inference Engine
    exec_net = iecore.load_network(network=net, device_name="CPU")

    print("IR successfully loaded into Inference Engine.")

    return


def main():
    args = get_args()
    load_to_IE(args.m)
    print("args: ", args)


if __name__ == "__main__":
    main()

"""
Run the following command inside /home/workspace to test all models

python3 feed_network.py -m models/human-pose-estimat
ion-0001.xml && python3 feed_network.py -m models/semantic-segmentation-adas-0001.xml && pytho
n3 feed_network.py -m models/text-detection-0004.xml && python3 feed_network.py -m models/vehi
cle-attributes-recognition-barrier-0039.xml
"""