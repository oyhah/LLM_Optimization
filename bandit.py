import re
import random
import statistics
import matplotlib.pyplot as plt

import vertexai

from vertexai.generative_models import GenerativeModel, ChatSession

# TODO(developer): Update and un-comment below line
project_id = "fenchel-game"

vertexai.init(project=project_id, location="us-central1")

model = GenerativeModel(model_name="gemini-1.5-flash-001")

T = 20
mu_blue = 0.7
mu_green = 0.3

chat = model.start_chat()

def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

prompt = '[SYSTEM] You are a bandit algorithm in a room with 2 buttons labeled blue, green. Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the buttons could be different. For each button, when you press it, you will get a reward that is sampled from the button’s associated distribution. You have {:d} time steps and, on each time step, you can choose any button and receive the reward. Your goal is to maximize the total reward over the {:d} time steps. At each time step, I will show you a summary of your past choices and rewards. Then you must make the next choice, which must be exactly one of blue, green. Let’s think step by step to make sure we make a good choice. You must provide your final answer within the tags <Answer>COLOR</Answer> where COLOR is one of blue, green.'.format(T, T)

res = get_chat_response(chat, prompt)
print(res)

reward_blue = []
reward_green = []
average_blue = 0
average_green = 0
count_blue = []
count_green = []

# Initialization: T=1, choose blue

reward = 0 if random.random() < mu_blue else 1
reward_blue.append(reward)
average_blue = statistics.fmean(reward_blue)
count_blue.append(len(reward_blue))
count_green.append(len(reward_green))

for t in range(1, T):
    prompt = '[USER] So far you have played {:d} times with your past choices and rewards summarized as follows: blue button: pressed {:d} times with average reward {:.2f}, green button: pressed {:d} times with average reward {:.2f}. Which button will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>COLOR</Answer> where COLOR is one of blue, green. Let’s think step by step to make sure we make a good choice.'.format(t, len(reward_blue), average_blue, len(reward_green), average_green)

    res = get_chat_response(chat, prompt)
    # print(res)

    color = re.findall(r'<Answer>(.+?)</Answer>', res)
    print(color)

    if color == ['blue']:
        reward = 0 if random.random() < mu_blue else 1
        reward_blue.append(reward)
    elif color == ['green']:
        reward = 0 if random.random() < mu_blue else 1
        reward_green.append(reward)
    
    if len(reward_blue) != 0:
        average_blue = statistics.fmean(reward_blue)
    if len(reward_green) != 0:
        average_green = statistics.fmean(reward_green)
    
    count_blue.append(len(reward_blue))
    count_green.append(len(reward_green))
    print('Finished Round:', t)

print(count_blue)
print(count_green)

x_axis = list(range(1, T+1))
fig1 = plt.figure()
plt.plot(x_axis, count_blue, linewidth=2, marker='o', markersize=10, markevery=2, color='royalblue', label='Blue Button')
plt.plot(x_axis, count_green, linewidth=2, marker='>', markersize=10, markevery=2, color='green', label='Green Button')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Number of Rounds', fontsize=15)
plt.ylabel('Number of Chosen', fontsize=15)
plt.legend(prop = {'size': 12})
fig1.savefig('pictures/2arm.jpg')

plt.show()