import streamlit as st
import replicate
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.spatial import ConvexHull

load_dotenv()

st.set_page_config(page_title="🖼️ Image Shape Classifier")

# Replicate API key setup
replicate_api = os.getenv('REPLICATE_API_TOKEN')
if not replicate_api:
    st.error('API key not found! Please make sure it is set in the .env file.')
else:
    st.sidebar.success('API key loaded successfully!', icon='✅')


# Functions for Shape Detection and Image Saving
def read_csv(csv_path):
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
        path_XYs = []
        unique_paths = np.unique(np_path_XYs[:, 0])
        for i in unique_paths:
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            XYs = []
            unique_shapes = np.unique(npXYs[:, 0])
            for j in unique_shapes:
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                XYs.append(XY)
            path_XYs.append(XYs)
        return path_XYs
    except Exception as e:
        print("An error occurred:", e)
        raise

def angle_between(v1, v2):
    dot_prod = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    return np.arccos(dot_prod / (mag_v1 * mag_v2))

def detect_and_replace_shapes(path_XYs):
    categorized_shapes = []
    for shapes in path_XYs:
        for XY in shapes:
            if len(XY) < 5:  # Too few points to form a complex shape
                continue
            
            # Circle Detection
            center = np.mean(XY, axis=0)
            radii = np.linalg.norm(XY - center, axis=1)
            radius_variance = np.std(radii)
            radius_mean = np.mean(radii)

            if radius_variance < 0.1 * radius_mean:
                # Create a perfect circle
                t = np.linspace(0, 2*np.pi, 100)
                perfect_circle = np.column_stack((center[0] + radius_mean * np.cos(t), center[1] + radius_mean * np.sin(t)))
                categorized_shapes.append(('Circle', perfect_circle))
                continue
            
            # Square/Rectangle Detection
            hull = ConvexHull(XY)
            if len(hull.vertices) == 4:
                angles = []
                for i in range(4):
                    v1 = XY[hull.vertices[i]] - XY[hull.vertices[i-1]]
                    v2 = XY[hull.vertices[i-2]] - XY[hull.vertices[i-1]]
                    angle = angle_between(v1, v2)
                    angles.append(np.degrees(angle))
                if all(85 < angle < 95 for angle in angles):
                    # Create a perfect square/rectangle
                    x_min, y_min = np.min(XY, axis=0)
                    x_max, y_max = np.max(XY, axis=0)
                    perfect_square = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]])
                    categorized_shapes.append(('Square/Rectangle', perfect_square))
                    continue
            
            # Fit a line to the points (for lines and near-lines)
            X = XY[:, 0].reshape(-1, 1)
            y = XY[:, 1]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            normalized_mse = mse / (np.ptp(y)**2)  # Normalized by the range of y values

            # Classification based on variance and MSE
            if normalized_mse < 0.05:
                # Create a straight line connecting initial and final points
                start_point = XY[0]
                end_point = XY[-1]
                perfect_line = np.array([start_point, end_point])
                categorized_shapes.append(('Line', perfect_line))
            elif normalized_mse < 0.1:
                categorized_shapes.append(('Near-Line', XY))
            else:
                categorized_shapes.append(('Doodle', XY))
    return categorized_shapes

def save_shapes_as_images(categorized_shapes, output_dir='output_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define unique shape types
    shape_types = list(set(shape_type for shape_type, _ in categorized_shapes))
    
    for shape_type in shape_types:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor('white')  # Set background color to white
        
        for s_type, XY in categorized_shapes:
            if s_type == shape_type:
                ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
                
        ax.set_aspect('equal')
        ax.axis('off')

        # Save each class as an image
        file_path = os.path.join(output_dir, f'{shape_type.replace("/", "_")}.png')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

        # Load the saved image and handle alpha channel
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img.shape[2] == 4:  # we have an alpha channel
            a1 = ~img[:, :, 3]  # extract and invert that alpha
            img = cv2.add(cv2.merge([a1, a1, a1, a1]), img)  # add up values (with clipping)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # strip alpha channel
        
        cv2.imwrite(file_path, img)


# Streamlit UI
st.title("CSV Shape Classifier and Image Processor")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Save uploaded CSV to a temporary location
    csv_path = f'temp_{uploaded_file.name}'
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the CSV to detect and save shapes
    with st.spinner('Processing CSV and detecting shapes...'):
        path_XYs = read_csv(csv_path)
        categorized_shapes = detect_and_replace_shapes(path_XYs)
        save_shapes_as_images(categorized_shapes)
    
    st.success("Shapes detected and saved successfully!")

    # Display images and classify them using Replicate
    output_dir = 'output_images'
    for shape_type in os.listdir(output_dir):
        image_path = os.path.join(output_dir, shape_type)
        
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=f"Loaded Image: {shape_type.split('.')[0]}", use_column_width=True)

            prompt_input = "The response should be only of single word ,The input is of roughly drawn shape or a doodle which does not resembles any shape in particular. Classify the roughly drawn shape into square/rectangle , circle , ellipse(circle which is slightly long horizontally or vertically) , triangle , star shape or noneoftheabove . Return the result in the json format with the shape name only even if the shapes are slightly skewed ."
            
            # Process the image using Replicate
            st.write("Processing the image...")

            with st.spinner("Classifying shape..."):
                output = replicate.run(
                    "lucataco/moondream2:72ccb656353c348c1385df54b237eeb7bfa874bf11486cf0b9473e691b662d31",
                    input={
                        "image": open(image_path, "rb"),
                        "prompt": prompt_input
                    }
                )

                # Convert the generator to a list to print and display
                output_list = list(output)
                filtered_output = [item for item in output_list if item.strip()]  # Remove empty strings

                st.success("Classification completed successfully!")

            st.subheader("Shape Classification Result")
            st.write(filtered_output)  # Displaying the results in Streamlit
            print(filtered_output)  # Printing the results in the terminal

else:
    st.info('Please upload a CSV file to start the process.')
