from image_utils import load_image , edge_detection
raw_data = load_image('CAT.jpg')
edges = edge_detection(raw_data)
