from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

# Function to break the video into images
def break_video_into_images(video_path, image_width, image_height):
    video = cv2.VideoCapture(video_path)
    #Frame Rate
    frame_rate=video.get(cv2.CAP_PROP_FPS)
    print("frame rate:",frame_rate)
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (image_width, image_height))
        image_path = f"image_{frame_count}.jpg"
        cv2.imwrite(image_path, resized_frame)
        frame_count += 1
    video.release()
    return frame_count
# Break the video into images
video_path="blood3.mp4"
frame_count = break_video_into_images(video_path, 300,300)


def plot_grayscale_histogram(image_path,output_path):
    # Open the image file
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Save the grayscale image
    grayscale_image.save(output_path)
    print(type(grayscale_image))

    
    # Get pixel values as a flattened array
    pixel_values = list(grayscale_image.getdata())
    
    # Plot histogram
    plt.hist(pixel_values, bins=256, range=(0, 256), density=True, color='grey', alpha =0.8)    
    # Set labels and title
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Grayscale Histogram')

    # Manually set the threshold value
    manual_threshold = 150

    image_array = np.array(grayscale_image)

    #Find rows and columns where pixel values are near zero or zero
    # rows = np.where(np.max(image_array == 0, axis=1))[0]
    # columns = np.where(np.max(image_array == 0, axis=0))[0]
    # print(rows)
    # print(len(rows))
    # print(columns)
    # print(len(columns))
    
    # Draw a vertical line at the threshold value
    plt.axvline(x=manual_threshold, color='red', linestyle='dashed', linewidth=2, label='Manual Threshold')
    
    # Show the plot
    plt.show()

def check_pixel(image_path,rows,columns):
    # Open the image file
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    image_array = np.array(grayscale_image)
    
    
    pixel_value = np.mean(image_array[rows,columns])
    print(pixel_value)
    return pixel_value


# Example usage
image_path = f"image_{frame_count-1}.jpg"
output_path = "greyscale.jpg"
#plot_grayscale_histogram(image_path,output_path)

def generate_list(value, num_elements):
    
    value = int(value)
    num_elements = max(num_elements, 1)
    
    if num_elements % 2 == 0:
        num_elements += 1
        
    num_before = (num_elements - 1) // 2
    num_after = (num_elements - 1) // 2

    new_list = list(range(value - num_before, value + num_after + 1))
    return new_list

#enter the value of pixel and no of rows or columns want to be generated
rows = generate_list(150, 10)
columns=generate_list(150, 10)
print(rows)
print(columns)


result=[]
for frame in range (frame_count):
    image_path = f"image_{frame}.jpg"
    p=check_pixel(image_path,rows,columns)
    result.append((frame,p))
    

# Create a DataFrame from the results
df = pd.DataFrame(result, columns=['Frame_Number', 'Pixel_Value'])

# Save the DataFrame to a CSV file
df.to_csv('pixel_values_for_row_column.csv', index=False)

# Extract the columns for frame number and pixel value
frame_number = df['Frame_Number']
pixel_value = df['Pixel_Value']

# Plot the data
plt.plot(frame_number, pixel_value)
plt.xlabel('Frame Number')
plt.ylabel('Pixel Value')
#plt.ylim(30,50)
#plt.xlim(0,1400)
plt.title('Pixel Value over Frame Number')


# Display the graph
plt.show()