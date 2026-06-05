import os
import joblib
import numpy as np
import warnings

PREMIUM_SCORE = {
    "rolls-royce": 10, "bentley": 10, "ferrari": 10, "lamborghini": 10,
    "aston martin": 9, "maserati": 9, "porsche": 9,
    "mercedes-benz": 8, "bmw": 8, "audi": 8, "tesla": 8,
    "jaguar": 8, "land rover": 8, "lexus": 8,
    "cadillac": 7, "lincoln": 7, "infiniti": 7, "acura": 7, "volvo": 7,
    "mini": 6, "buick": 6, "volkswagen": 6,
    "toyota": 5, "honda": 5, "mazda": 5, "subaru": 5,
    "hyundai": 4, "kia": 4, "nissan": 4, "chevrolet": 4,
    "ford": 4, "jeep": 4, "gmc": 4,
    "dodge": 3, "chrysler": 3, "mitsubishi": 3, "suzuki": 3,
    "pontiac": 3, "mercury": 3, "saturn": 3, "scion": 3, "saab": 3,
    "daewoo": 2, "geo": 2, "oldsmobile": 2, "plymouth": 2,
}

SUV_PATTERN    = r'suv|explorer|tahoe|suburban|x5|q5|q7|rav4|cr-v'
SPORTS_PATTERN = r'm3|m5|amg|rs|corvette|mustang|challenger|911'
EV_PATTERN     = r'hybrid|electric|ev|plug-in'


def _age_bucket(car_age: int) -> str:
    if car_age <= 3:   return 'new'
    if car_age <= 7:   return 'midsized'
    if car_age <= 12:  return 'old'
    if car_age <= 20:  return 'very_old'
    if car_age <= 25:  return 'floor'
    return 'vintage'


class LiveAuctionAgent:
    def __init__(self):
        self.bankroll        = 500_000.0
        self.predicted_value = 0.0
        self.round_number    = 0

        # per-auction bid tracking
        self.last_own_bid    = 0.0  
        self.last_rival_bid  = 0.0 
        self.rival_jumps     = []  

        base_path = os.path.dirname(os.path.abspath(__file__))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model   = joblib.load(os.path.join(base_path, "model_AyushRaigandhi.pkl"))
            mappings     = joblib.load(os.path.join(base_path, "encoders_AyushRaigandhi.pkl"))

        self.state_avg_mapping = mappings['state_avg_mapping']
        self.brand_avg_mapping = mappings['brand_avg_mapping']
        self.model_popularity  = mappings['model_popularity']
        self.make_freq         = mappings['make_freq']
        self.global_mean       = mappings['global_mean']
        self.feature_columns   = mappings['feature_columns']

    # Feature Engineering

    def _build_features(self, f: dict) -> dict:
        import re

        year         = int(f.get('year', 2012))
        condition    = float(f.get('condition', 3.60))
        odometer     = float(f.get('odometer', 12100.00))
        make         = str(f.get('make',         'unknown')).lower().strip()
        model        = str(f.get('model',        'unknown')).lower().strip()
        trim         = str(f.get('trim',         'unknown')).lower().strip()
        body         = str(f.get('body',         'unknown')).lower().strip()
        transmission = str(f.get('transmission', 'unknown')).lower().strip()
        state        = str(f.get('state',        'unknown')).lower().strip()
        color        = str(f.get('color',        'unknown')).lower().strip()
        interior     = str(f.get('interior',     'unknown')).lower().strip()

        car_age         = max(1, 2026 - year)
        odometer_log    = np.log1p(odometer)
        usage_intensity = odometer / car_age
        overused        = bool(usage_intensity > 25_000)
        is_vintage      = bool(car_age >= 30)
        deprecated      = bool(condition <= 2.0)
        age_buck        = _age_bucket(car_age)
        years_to_floor  = max(0, 25 - car_age)
        premium         = PREMIUM_SCORE.get(make, 5)
        is_suv          = int(bool(re.search(SUV_PATTERN,    model, re.I)))
        is_sports       = int(bool(re.search(SPORTS_PATTERN, model, re.I)))
        is_hybrid_ev    = int(bool(re.search(EV_PATTERN,     model, re.I)))
        rare_make       = int(self.make_freq.get(make, 0) < 10)
        model_pop       = self.model_popularity.get(model, 0)
        state_avg       = self.state_avg_mapping.get(state, self.global_mean)
        brand_avg       = self.brand_avg_mapping.get(make,  self.global_mean)

        return {
            'year': year, 'condition': condition,
            'make': make, 'model': model, 'trim': trim, 'body': body,
            'transmission': transmission, 'state': state,
            'color': color, 'interior': interior,
            'odometer_log': odometer_log, 'car_age': car_age,
            'usage_intensity': usage_intensity, 'overused': overused,
            'is_vintage': is_vintage, 'age_bucket': age_buck,
            'years_until_floor': years_to_floor, 'premium_score': premium,
            'is_suv': is_suv, 'is_sports': is_sports, 'is_hybrid_ev': is_hybrid_ev,
            'rare_make': rare_make, 'model_popularity': model_pop,
            'deprecated': deprecated,
            'state_avg_price': state_avg, 'brand_avg_price': brand_avg,
        }

    # Prediction  

    def analyze_item(self, item_features: dict):
        import pandas as pd
        features_dict = self._build_features(item_features)
        input_df      = pd.DataFrame([features_dict])[self.feature_columns]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_pred = self.model.predict(input_df)[0]

        self.predicted_value = float(np.expm1(log_pred))

        # reset all per-auction state
        self.round_number   = 0
        self.last_own_bid   = 0.0
        self.last_rival_bid = 0.0
        self.rival_jumps    = []

    #  Bidding Strategy 
    #
    #  Three signals computed each round:
    #
    #  1. DEAL QUALITY  = (predicted - current_bid) / predicted
    #     How far below our valuation the bid currently sits.
    #     High → cheap car, bid boldly. Low → nearly at ceiling, bid tightly.
    #
    #  2. RIVAL AGGRESSION = rival's last jump / predicted_value
    #     If the rival is making large jumps they're confident; back off.
    #     If their jumps are tiny they're weak; one assertive raise can end it.
    #
    #  3. BID-TO-VALUE RATIO = current_bid / predicted_value
    #     Where in the price range we're sitting. Used to enter early / exit
    #     gracefully without hard-stopping exactly at ceiling.
    #
    #  These three signals combine into a context multiplier that scales
    #  the base geometric-decay aggression up or down each round.

    MARGIN_FACTOR   = 0.85
    BANKROLL_CAP    = 0.15
    BASE_AGGRESSION = 0.30   
    DECAY_RATE      = 0.88  
    MIN_INCREMENT   = 100.0  

    def place_bid(self, current_highest_bid: float) -> float:
        if self.bankroll <= 0:
            return 0.0

        self.round_number += 1

        # Ceilings
        value_ceiling    = self.predicted_value * self.MARGIN_FACTOR
        bankroll_ceiling = self.bankroll        * self.BANKROLL_CAP
        ceiling          = min(value_ceiling, bankroll_ceiling)

        if current_highest_bid >= ceiling:
            return 0.0

        #  Signal 1: Deal Quality
        deal_quality = (self.predicted_value - current_highest_bid) / max(self.predicted_value, 1)
        deal_quality = max(0.0, min(1.0, deal_quality))

        #  Signal 2: Rival Aggression
        if self.last_own_bid > 0 and current_highest_bid > self.last_own_bid:
            rival_increment = current_highest_bid - self.last_own_bid
            self.rival_jumps.append(rival_increment)

        if self.rival_jumps:
            avg_rival_jump   = np.mean(self.rival_jumps)
            rival_aggression = avg_rival_jump / max(self.predicted_value, 1)
            rival_factor = 1.0 - np.clip(rival_aggression * 8, 0.0, 0.4)
        else:
            rival_factor = 1.0   

        #  Signal 3: Bid-to-Value Ratio 
        bid_ratio = current_highest_bid / max(self.predicted_value, 1)
        
        if bid_ratio < 0.40:
            position_factor = 1.35  
        elif bid_ratio < 0.60:
            position_factor = 1.10  
        elif bid_ratio < 0.75:
            position_factor = 0.90  
        else:
            position_factor = 0.65  

        # Combine signals into final increment 
        base_decay  = self.BASE_AGGRESSION * (self.DECAY_RATE ** (self.round_number - 1))
        context_k   = base_decay * rival_factor * position_factor * (0.5 + deal_quality)
        headroom    = ceiling - current_highest_bid
        increment   = max(self.MIN_INCREMENT, headroom * context_k)

        next_bid = current_highest_bid + increment
        next_bid = min(next_bid, ceiling, self.bankroll)

        if next_bid <= current_highest_bid:
            return 0.0

        self.last_own_bid = round(next_bid, 2)
        return self.last_own_bid


    def auction_result(self, won, winning_bid, actual_price, current_bankroll):
        if won:
            self.bankroll = max(0.0, self.bankroll - winning_bid)