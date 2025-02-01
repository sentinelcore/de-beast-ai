import tweepy
import openai
import numpy as np
import gymnasium as gym
import shimmy
from fastapi import FastAPI
from stable_baselines3 import PPO
import uvicorn
import os
from mangum import Mangum

# Twitter API Keys (Replace with your keys)
TWITTER_API_KEY = "your_twitter_api_key"
TWITTER_API_SECRET = "your_twitter_api_secret"
TWITTER_ACCESS_TOKEN = "your_access_token"
TWITTER_ACCESS_SECRET = "your_access_secret"
OPENAI_API_KEY = "your_openai_api_key"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)

# Authenticate OpenAI API
openai.api_key = OPENAI_API_KEY

app = FastAPI()

# AI-Generated Tweet Function
def generate_tweet():
    prompt = "Generate an engaging tweet from an EV charger about its revenue, charging discounts, or referral system."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

@app.get("/tweet")
def post_ai_tweet():
    tweet = generate_tweet()
    api.update_status(tweet)
    return {"Status": "Tweet posted", "Content": tweet}

# AI-Powered Pricing Model
class ChargingPriceEnv(gym.Env):
    def __init__(self):
        super(ChargingPriceEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # 5 different price levels
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset function now accepts a seed argument for compatibility."""
        super().reset(seed=seed)  # Call parent reset to handle seeding
        self.state = np.array([50, 20, 5], dtype=np.float32)  # Initial values
        return self.state, {}

    def step(self, action):
        demand, energy_cost, competitors = self.state
        new_price = action * 10 + 50
        revenue = demand * new_price

        # Simulate demand fluctuations
        demand_change = -1 * (new_price / 100) * np.random.randint(1, 10)
        demand += demand_change
        energy_cost += np.random.randint(-5, 5)
        
        reward = revenue - energy_cost

        # Update state
        self.state = np.array([demand, energy_cost, competitors], dtype=np.float32)
        return self.state, reward, False, {}, {}

env = ChargingPriceEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("charging_pricing_model")

@app.get("/get_price")
def get_optimal_price():
    model = PPO.load("charging_pricing_model")

    # Gymnasium returns (observation, info_dict), we only need observation
    obs, _ = env.reset()  # Extract obs from tuple

    action, _states = model.predict(obs)
    optimal_price = action * 10 + 50

    return {"Optimal Charging Price": f"${optimal_price}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

handler = Mangum(app)
