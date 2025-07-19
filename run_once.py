# run_once.py
import time
from recommender_ws import seed_initial_prices, backtest_probabilities, recommend, CANDIDATES

if __name__ == "__main__":
    seed_initial_prices()
    prob_map = backtest_probabilities(CANDIDATES, samples=100)
    time.sleep(30)
    recommend(prob_map, top_n=5)
