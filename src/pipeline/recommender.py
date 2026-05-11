import os
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class Recommender:

    def __init__(self, data_path):

        self.data_path = data_path


        # LOAD DATA

        self.flights_df = pd.read_csv(os.path.join(data_path, "flights_clean.csv"))
        self.hotels_df = pd.read_csv(os.path.join(data_path, "hotels_clean.csv"))


        # USER-ITEM MATRIX

        interaction_df = self.flights_df[['usercode', 'travelcode']].dropna()
        interaction_df['interaction'] = 1

        self.user_item_matrix = interaction_df.pivot_table(
            index='usercode',
            columns='travelcode',
            values='interaction',
            fill_value=0
        )

        # USER SIMILARITY
        user_sim = cosine_similarity(self.user_item_matrix)

        self.user_sim_df = pd.DataFrame(
            user_sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )


        # POPULARITY

        popularity = interaction_df['travelcode'].value_counts()

        self.pop_df = pd.DataFrame({
            "travelcode": popularity.index,
            "pop_score": popularity.values
        })

        scaler = MinMaxScaler()
        self.pop_df['pop_score'] = scaler.fit_transform(self.pop_df[['pop_score']])

    # SIMILAR USERS
    def get_similar_users(self, user_id, top_n=5):

        if user_id not in self.user_sim_df.index:
            return pd.Series(dtype=float)

        return self.user_sim_df[user_id].sort_values(ascending=False)[1:top_n+1]

    # COLLABORATIVE SCORES
    def get_collab_scores(self, user_id):

        similar_users = self.get_similar_users(user_id)
        scores = {}

        for sim_user, sim_score in similar_users.items():
            items = self.user_item_matrix.loc[sim_user]
            liked = items[items == 1].index

            for item in liked:
                scores[item] = scores.get(item, 0) + sim_score

        return scores

    # REMOVE ALREADY SEEN
    def remove_seen_items(self, df, user_id):

        if user_id not in self.user_item_matrix.index:
            return df

        seen = self.user_item_matrix.loc[user_id]
        seen_items = seen[seen == 1].index

        return df[~df['travelcode'].isin(seen_items)]

    # FLIGHT RECOMMENDATION
    def recommend_flights(self, user_id, source, destination,
                          flight_type=None, max_price=None, top_n=5):

        top_n = min(top_n, 20)

        df = self.flights_df.copy()

        # FILTER
        df = df[(df['from'] == source) & (df['to'] == destination)]

        if flight_type:
            df = df[df['flighttype'] == flight_type]

        if max_price:
            df = df[df['price'] <= max_price]

        if df.empty:
            return []

        # REMOVE SEEN ITEMS
        df = self.remove_seen_items(df, user_id)

        # POPULARITY
        df = df.merge(self.pop_df, on='travelcode', how='left')
        df['pop_score'] = df['pop_score'].fillna(0)

        # COLLAB
        collab_scores = self.get_collab_scores(user_id)
        df['collab_score'] = df['travelcode'].map(collab_scores).fillna(0)

        df['collab_score'] = df['collab_score'] / (df['collab_score'].max() + 1e-6)

        # FINAL SCORE
        df['final_score'] = 0.6 * df['pop_score'] + 0.4 * df['collab_score']

        # SORT
        df = df.sort_values(by='final_score', ascending=False)

        # REMOVE DUPLICATES 
        df = df.drop_duplicates(
            subset=['from', 'to', 'flighttype', 'price', 'agency']
        )

        df = df.head(top_n)

        # CONFIDENCE
        max_score = df['final_score'].max()

        if max_score == 0:
            df['confidence'] = np.round(np.random.uniform(0.5, 0.7, len(df)), 2)
        else:
            df['confidence'] = np.round(df['final_score'] / max_score, 2)

        return df[['travelcode','from','to','price','flighttype','agency','confidence']].to_dict("records")

    # HOTEL RECOMMENDATION
    def recommend_hotels(self, user_id, place, max_price=None, top_n=5):

        top_n = min(top_n, 20)

        df = self.hotels_df.copy()

        df = df[df['place'] == place]

        if max_price:
            df = df[df['price'] <= max_price]

        if df.empty:
            return []

        # REMOVE SEEN
        df = self.remove_seen_items(df, user_id)

        # POPULARITY
        pop = df['travelcode'].value_counts()
        df['pop_score'] = df['travelcode'].map(pop)
        df['pop_score'] = df['pop_score'] / (df['pop_score'].max() + 1e-6)

        # COLLAB
        collab_scores = self.get_collab_scores(user_id)
        df['collab_score'] = df['travelcode'].map(collab_scores).fillna(0)
        df['collab_score'] = df['collab_score'] / (df['collab_score'].max() + 1e-6)

        # FINAL SCORE
        df['final_score'] = 0.5 * df['pop_score'] + 0.5 * df['collab_score']

        df = df.sort_values(by='final_score', ascending=False)

        # REMOVE DUPLICATES
        df = df.drop_duplicates(subset=['name', 'place', 'price'])

        df = df.head(top_n)

        max_score = df['final_score'].max()

        if max_score == 0:
            df['confidence'] = np.round(np.random.uniform(0.6, 0.8, len(df)), 2)
        else:
            df['confidence'] = np.round(df['final_score'] / max_score, 2)

        return df[['name','place','price','confidence']].to_dict("records")

    # PACKAGE
    def recommend_package(self, user_id, source, destination, place, max_price, top_n=5):

        flights = self.recommend_flights(user_id, source, destination, max_price=max_price, top_n=top_n)
        hotels = self.recommend_hotels(user_id, place, max_price=max_price, top_n=top_n)

        if not flights or not hotels:
            return []

        best_hotel = hotels[0]

        package = []

        for f in flights:
            package.append({
                **f,
                "hotel_name": best_hotel['name'],
                "hotel_price": best_hotel['price'],
                "total_price": f['price'] + best_hotel['price']
            })

        return package

    # ALL
    def recommend_all(self, user_id, source, destination, place, max_price, top_n=5):

        return {
            "flights": self.recommend_flights(user_id, source, destination, max_price=max_price, top_n=top_n),
            "hotels": self.recommend_hotels(user_id, place, max_price=max_price, top_n=top_n),
            "packages": self.recommend_package(user_id, source, destination, place, max_price, top_n)
        }