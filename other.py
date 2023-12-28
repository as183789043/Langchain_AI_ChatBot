from langchain.memory import ConversationBufferWindowMemory
from skimage import io
from openai import OpenAI
import os 


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])




def generate_image(response,image_model_name:str):
    client = OpenAI()

    response = client.images.generate(
      model=image_model_name,
      prompt=f"{response} 依照上述內容挑選最重要的內容生成貓咪漫畫，畫面不能有裁切",
      size="1792x1024",
      quality="hd",
      n=1,
    )

    return response.data[0].url



def display_image(image_url):
    image = io.imread(image_url)
    io.imshow(image)
    return io.show()



if __name__=="__main__":
    image_url=generate_image("車禍發生時，車主應立即停車，打開雙黃燈，並下車拿三角警示牌立在車後方30公尺至100公尺處")
    display_image(image_url)