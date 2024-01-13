
import argparse
import cv2
import numpy as np

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

def main():
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter,
                                     description="Generate a .png image of a specified maker.")
    
    parser.add_argument('--id', 
                        default=1, 
                        type=int,
                        help='Marker id to generate')
    
    parser.add_argument('--size', 
                        default=200, 
                        type=int,
                        help='Side length in pixels')
    
    dict_options = [s for s in dir(cv2.aruco) if s.startswith("DICT")]
    option_str = ", ".join(dict_options)
    dict_help = "Dictionary to use. Valid options include: {}".format(option_str)

    parser.add_argument('--dictionary',
                        default="DICT_6X6_250", 
                        type=str, 
                        choices=dict_options,
                        help=dict_help, 
                        metavar='')
    
    args, unknown = parser.parse_known_args()

    dictionary_id = cv2.aruco.__getattribute__(args.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    image = cv2.aruco.generateImageMarker(dictionary, args.id, args.size)

    # Add a white border to the image
    border_size = 20
    border_color = [255, 255, 255]
    image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, 
                               cv2.BORDER_CONSTANT, value=border_color)

    cv2.imwrite("marker_{:04d}.png".format(args.id), image)

if __name__ == "__main__":
    main()




    # image = np.zeros((args.size, args.size), dtype=np.uint8)
    # image = cv2.aruco.generateImageMarker(dictionary, args.id, args.size, image, 1)
    # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

'''

    # aruco_generate_marker_node = Node(
    #     package='ros2_aruco',
    #     executable='aruco_generate_marker',
    #     name='aruco_generate_marker'
    # )
    


        <!-- AR marker -->
        <include>
        <uri>model://marker26_8cm</uri>
        <pose>1.4 0 1.5 0 1.5707 0</pose>
        </include>
        
'''