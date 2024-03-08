import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

def answer_image(question, img_list):
    print(question)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    def get_answer(image, text):
        try:

            img = Image.open(BytesIO(image)).convert("RGB")


            encoding = processor(img, text, return_tensors="pt")


            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = model.config.id2label[idx]

            return answer

        except Exception as e:
            return str(e)

    # Convert the list to a NumPy array
    img_array = np.array(img_list, dtype=np.uint8)
    cv2.imwrite("saveeeeed.jpg", img_array)
    # Reshape the array to match the image dimensions
    width, height, channels = img_array.shape

    # Create a PIL Image from the NumPy array
    uploaded_file = Image.fromarray(img_array.reshape((height, width, channels)))




    if uploaded_file and question is not None:

        image = uploaded_file
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='JPEG')
        image_bytes = image_byte_array.getvalue()


        answer = get_answer(image_bytes, question)


        return(answer)


#image = Image.open("dogcat.jpg")
#img_list = list(image.getdata())
#width, height = image.size
#
## Reshape the 2D list to a 3D list
#img_3d_list = [img_list[i * width:(i + 1) * width] for i in range(height)]
#print(answer_image("How many cats are there", img_3d_list))