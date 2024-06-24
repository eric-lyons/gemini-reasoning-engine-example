import os
import requests
from dotenv import load_dotenv
from google.cloud import speech
import yfinance as yf
from openai.error import InvalidRequestError

import vertexai

from vertexai.preview.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Tool,
    Part
)
import vertexai.preview.generative_models as generative_models
from vertexai.preview import reasoning_engines


# Load environment variables from .flaskenv
load_dotenv()

project = os.environ['PROJECT']
region = os.environ['REGION']
model_selector = os.environ['MODEL']
temp = os.environ['TEMPERATURE']
lan = os.environ['LANGUAGE']
max_tokens = os.environ['MAX_TOKENS']
weather_api_key = os.environ['WEATHER_API_KEY']
key_path = os.environ['KEY_PATH']
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path


def generate_ai_chatbot_response(messages):
    # TO DO: Add prompt as os variable.
    try:
        new_messages = str(messages)
        prompt = '''{}'''.format(new_messages)
        # reply = generate_ai_chatbot_response2(prompt)
        reply = get_chat_response_function(prompt)
        messages.append({"role": "system", "content": reply})
        return "success", messages
    except InvalidRequestError as e:
        return "fail", str(e)


def generate_ai_chatbot_response2(prompt):

    # TO DO: Make this a function call
    # TO DO: Make this multi-turn.
    vertexai.init(project=project, location=region)
    model = GenerativeModel(model_selector)
    generation_config = {
        "max_output_tokens": int(max_tokens),
        "temperature": float(temp),
        "top_p": 1, }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    try:
        responses = model.generate_content(
            [prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,)
        gen_string = []
        for chunk in responses:
            gen_string.append(chunk.text)
        gen_string = ",".join(str(element) for element in gen_string)
        # Can improve formatting here:
        reply = gen_string.replace('\n', ' ')
        return reply

    except InvalidRequestError as e:
        return "fail", str(e)


# Speech-to-text Documentation
# https://codelabs.developers.google.com/codelabs/cloud-speech-text-python3#3


def speech_to_text_gcs(uri):
    # TO DO: Add try catch error handling
    uri = uri
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        language_code=lan,
    )
    audio = speech.RecognitionAudio(uri=uri,)
    # Synchronous speech recognition request
    try:
        response = client.recognize(config=config, audio=audio)
        for result in response.results:
            text = result.alternatives[0].transcript

            return text
    except InvalidRequestError as e:
        return "fail", str(e)


def get_stock_price(ticker: str):
    '''Get the current stock price of a given company
    based on stock ticker or company name e.g.
    What is the stock price of Google?'''
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if not hist.empty:
        close = (str(hist['Close']))
        api_response = "price: {}".format(close)
        return api_response
    else:
        return {"error": "No data available"}


def get_current_weather(location: str):
    '''Get the current weather in a given location.
    Use the location like city name to get the information
    E.G. What is the weather in New York?'''
    api_key = weather_api_key
    # GET request to the weatherapi with the provided location
    url = "http://api.weatherapi.com/v1/current.json?key={}&q={}&aqi=no".format(api_key, location)
    response = requests.get(url)

    # Convert the response to a JSON object
    response = response.json()
    condition = response['current']['condition']['text']
    weather_temp = str(response['current']['temp_f'])
    api_response = '''The current weather in {} is
    {} with a temperature of {} F.'''.format(location, condition, weather_temp)
    return api_response


def get_chat_response_function(prompt):
    model = "gemini-1.5-pro-preview-0409"
    vertexai.init(project=project, location=region)
    agent = reasoning_engines.LangchainAgent(
        model=model,
        tools=[get_current_weather, get_stock_price],
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    prompt = prompt
    response = agent.query(input=prompt)
    response = response.get("output")
    return response


# def get_chat_response_function(prompt):
#     prompt = prompt
#     get_stock_price_func = FunctionDeclaration(
#             name="get_stock_price",
#             description='''Get the current stock price of a company based
#             on stock ticker or company name
#             e.g. What is the stock price of Google?''',
#             parameters={
#                 "type": "object",
#                 "properties": {
#                     "ticker": {
#                         "type": "string",
#                         "description": "Stock ticker symbol"
#                     }
#                 }
#             },
#         )

#     get_current_weather_func = FunctionDeclaration(
#         name="get_current_weather",
#         description='''Get the current weather in a given location.
#             Use the location like city name to get the information
#             E.G. What is the weather in New York?''',
#         parameters={
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": '''The location to get the weather for.
#                     For example, 'New York, NY' or London, UK''',
#                 },
#             },
#             "required": ["location"],
#                 },

#     )
#     tools = Tool(
#         function_declarations=[
#             # google_search_func,
#             get_current_weather_func,
#             get_stock_price_func,
#             ],
#      )
#     # Model Initialization
#     model2 = GenerativeModel(model_selector,
#                              generation_config={"temperature": float(temp)},
#                              tools=[tools],)

#     model3 = GenerativeModel(model_selector,
#                              generation_config={"temperature": float(temp)})

#     chat = model2.start_chat(response_validation=False)
#     response = chat.send_message(prompt)
#     response = response.candidates[0].content.parts[0]
#     print("this is a response... ")
#     print(response)
#     # function_name_list = ['get_stock_price','get_current_weather']
#     # move to not in list operator below
#     if response.function_call.name == "get_stock_price" or response.function_call.name == "get_current_weather":
#         api_requests_and_responses = []
#         function_calling_in_process = True
#         while function_calling_in_process:
#             try:
#                 print("fuction calling.." + response.function_call.name)
#                 params = {}
#                 # Extract the function call parameters from the response
#                 for key, value in response.function_call.args.items():
#                     params[key] = value
#                 print(response.function_call.name)
#                 if response.function_call.name == "get_current_weather":
#                     location = params["location"]
#                     api_response = get_current_weather(location)
#                     api_requests_and_responses.append(
#                         [response.function_call.name, params, api_response])

#                 if response.function_call.name == "get_stock_price":
#                     ticker = params["ticker"]
#                     api_response = get_stock_price(ticker)
#                     api_requests_and_responses.append(
#                         [response.function_call.name,
#                             params, api_response])

#                 response = chat.send_message(
#                     Part.from_function_response(
#                         name=response.function_call.name,
#                         response={"content": api_response, }, ), )
#                 response = response.candidates[0].content.parts[0].text
#                 return response

#             except AttributeError:
#                 function_calling_in_process = False
#                 return "these are not the droids you are looking for"
#     # Generative Fall back logic here:
#     else:
#         print("no function prompt")
#         chat = model3.start_chat(response_validation=False)
#         response = chat.send_message(prompt)
#         response = response.candidates[0].content.parts[0].text
#         response = '''This app is designed specifically for
#         stock prices and weather, but here is
#         your answer: {} If this question was about stocks or
#         weather, please try rephrasing it.'''.format(response)
#         return response
